# -*- coding: utf-8 -*-
# pip uninstall -y unsloth xformers trl peft accelerate bitsandbytes datasets transformers gradio ipdb torch wandb
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
# pip install torch==2.3.0 gradio ipdb wandb

# * | // MARK: import =====================================
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from threading import Thread
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextIteratorStreamer
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import gradio as gr
import ipdb as pdb
import torch
import wandb

# * | // MARK: config =====================================
# unsloth supports RoPE (Rotary Position Embeddings), which scales internally
MAX_SEQ_LENGTH = 4096
COMPUTE_DTYPE = (
    torch.bfloat16
)  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+, A6000 is Ampere architecture.
MODEL_DTYPE_4BIT = True
# 4bit pre quantized models from unsloth (low network traffic, and no need to do the quantization before training the lora).
MODELS_4BIT = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit",
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",
]
HF_TOKEN = "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"
TRAIN_PER_DEVICE_TRAIN_BATCH_SIZE = 4
TRAIN_PER_DEVICE_EVAL_BATCH_SIZE = 1
TRAIN_EVALUATION_STRATEGY = "steps"
TRAIN_EVAL_STEPS = 0.1
TRAIN_GRADIENT_ACCUMULATION_STEPS = 4
TRAIN_WARMUP_RATIO = 0.2
TRAIN_NUM_TRAIN_EPOCHS = 8
TRAIN_LEARNING_RATE = 1e-4
TRAIN_FP16 = not is_bfloat16_supported()
TRAIN_BF16 = is_bfloat16_supported()
TRAIN_LOGGING_STEPS = 10
TRAIN_OPTIM = "adamw_8bit"
TRAIN_WEIGHT_DECAY = 0.01
TRAIN_LR_SCHEDULER_TYPE = "warmup_stable_decay"
# /home/kwb425/Applications/miniconda3/envs/1/lib/python3.9/site-packages/transformers/optimization.py:410
TRAIN_LR_SCHEDULER_KWARGS = {"num_stable_steps": 30, "num_decay_steps": 60, "min_lr_ratio": 1e-5, "num_cycles": 0.5}
TRAIN_SEED = 3407
TRAIN_OUTPUT_DIR = "output"

# * | // MARK: load model =====================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=COMPUTE_DTYPE,
    load_in_4bit=MODEL_DTYPE_4BIT,
    # token = HF_TOKEN # use one if using gated models like meta-llama/Llama-2-7b-hf
)

# * | // MARK: lora =====================================
# peft, updating 1 to 10% of all parameters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # rank, 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # Supports rank stabilized LoRA
    loftq_config=None,  # Supports LoftQ
)


# * | // MARK: data prep =====================================
""" 
1. unsloth relies on ShareGPT style 
    when data is in this form
        `[{"from": "human", "value" : "Hi"}, {"from": "gpt", "value" : "Hi"}]`
            we need mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
    when data is in this form
        `[{"role": "user", "content" : "Hi"}, {"role": "assistant", "content" : "Hi"}]`
            we need no mapping
2. after the optional mapper applied, we use `get_chat_template` to convert it to `llama-3 style`
    ```
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    Hey there! How are you?<|eot_id|><|start_header_id|>user<|end_header_id|>
    I'm great thanks!<|eot_id|>
    ```
    * note that there is no closing tag for <|begin_of_text|>
"""
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",  # zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    # mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)


def formatting_prompts_func(whole_dataset):
    conversations = whole_dataset["conversations"]
    # `texts` is now in `llama-3 style`
    texts = [
        tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        for conversation in conversations
    ]
    return {
        "text": texts,
    }


dataset = load_dataset("NoNameFactory/synth600", split="train")
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    batch_size=1,
)
# dataset = dataset.select(range(N))
dataset = dataset.train_test_split(test_size=0.05)

# * | // MARK: collator (masking) =====================================
# To train only on completions (ignoring the user's input) read TRL's docs (https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only).
response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
instruction_template = "<|start_header_id|>user<|end_header_id|>\n\n"
collator = DataCollatorForCompletionOnlyLM(
    instruction_template=instruction_template,
    response_template=response_template,
    tokenizer=tokenizer,
)


# * | // MARK: train =====================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=TRAIN_PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_PER_DEVICE_EVAL_BATCH_SIZE,
        evaluation_strategy=TRAIN_EVALUATION_STRATEGY,
        eval_steps=TRAIN_EVAL_STEPS,
        gradient_accumulation_steps=TRAIN_GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=TRAIN_WARMUP_RATIO,
        num_train_epochs=TRAIN_NUM_TRAIN_EPOCHS,
        learning_rate=TRAIN_LEARNING_RATE,
        fp16=TRAIN_FP16,
        bf16=TRAIN_BF16,
        logging_steps=TRAIN_LOGGING_STEPS,
        optim=TRAIN_OPTIM,
        weight_decay=TRAIN_WEIGHT_DECAY,
        lr_scheduler_type=TRAIN_LR_SCHEDULER_TYPE,
        lr_scheduler_kwargs=TRAIN_LR_SCHEDULER_KWARGS,
        seed=TRAIN_SEED,
        output_dir=TRAIN_OUTPUT_DIR,
    ),
    data_collator=collator,
)

# * | // MARK: logging =====================================
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved, training starting...")
trainer_stats = trainer.train()  # Training fire away!
wandb.finish()
print(f"finish training! gathering stats...")
model.config.use_cache = True
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# * | // MARK: save & load =====================================
# * Merge to 16bit, for later used with UNSLOTH or HUGGINGFACE
# model.save_pretrained_merged(
#     "model",
#     tokenizer,
#     save_method="merged_16bit",
# )
# model.push_to_hub_merged(
#     "hf/model",
#     tokenizer,
#     save_method="merged_16bit",
#     token=HF_TOKEN,
# )
# * Save to 8bit q8_0, for later used with GGUF (c/c++, https://github.com/ggerganov/llama.cpp)
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# model.save_pretrained_gguf(
#     "model",
#     tokenizer,
# )
# model.push_to_hub_gguf("hf/model", tokenizer, token=HF_TOKEN)
# * Merge to 4bit, for later used with UNSLOTH or HUGGINGFACE
# model.save_pretrained_merged(
#     "model",
#     tokenizer,
#     save_method="merged_4bit",
# )
# model.push_to_hub_merged(
#     "hf/model",
#     tokenizer,
#     save_method="merged_4bit",
#     token=HF_TOKEN,
# )
# * Save to 16bit, for later used with GGUF (c/c++, https://github.com/ggerganov/llama.cpp)
# model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
# model.push_to_hub_gguf("hf/model", tokenizer, quantization_method="f16", token=HF_TOKEN)
# * Just LoRA adapters, for later used with UNSLOTH or HUGGINGFACE
# model.save_pretrained_merged(
#     "model",
#     tokenizer,
#     save_method="lora",
# )
# model.push_to_hub_merged(
#     "hf/model",
#     tokenizer,
#     save_method="lora",
#     token=HF_TOKEN,
# )
# * Save to q4_k_m, for later used with GGUF (c/c++, https://github.com/ggerganov/llama.cpp)
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
# model.push_to_hub_gguf(
#     "hf/model", tokenizer, quantization_method="q4_k_m", token=HF_TOKEN
# )
# * Saving only lora (using unsloth class)
model.push_to_hub(
    "NoNameFactory/llama-3.1-8b-it-4bit-synth600",
    token=HF_TOKEN,
)
# * Loading only lora (using unsloth class)
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="NoNameFactory/llama-3.1-8b-it-4bit-synth600",  # Lora
#     max_seq_length=MAX_SEQ_LENGTH,
#     dtype=COMPUTE_DTYPE,
#     load_in_4bit=MODEL_DTYPE_4BIT,
# )
# * Loading only lora (using huggingface class)
# model = AutoPeftModelForCausalLM.from_pretrained(
#     "NoNameFactory/llama-3.1-8b-it-4bit-synth600",  # Lora
#     load_in_4bit=MODEL_DTYPE_4BIT,
# )
# tokenizer = AutoTokenizer.from_pretrained(
#     "NoNameFactory/llama-3.1-8b-it-4bit-synth600"
# )

# * | // MARK: inference =====================================
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
# we will use `from` `value` style, so mapping comes in as dict.
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",  # zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)


def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    messages = [
        {
            "from": "system",
            "value": "You are a helpful AI assistant. Please follow the user's request kindly.",
        }
    ]
    messages += [
        {"from": "human", "value": item[0]} for item in history_transformer_format
    ]
    messages += [
        {"from": "gpt", "value": item[1]} for item in history_transformer_format
    ]
    messages[-1]["value"] = ""
    # Prepare model inputs
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
    ).to("cuda")

    print(f"message: ${message}")
    print(f"history: ${history}")
    print(f"model_inputs: ${model_inputs}")
    # Initialize the text streamer
    text_streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )

    # Set up generation arguments
    generate_kwargs = dict(
        input_ids=model_inputs,
        streamer=text_streamer,
        max_new_tokens=10000,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # offloading to GPU, if CPU is used, then this will suffer from GIL
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # cpu streamer, this will wait until gpu thread emits each chunk
    partial_message = ""
    for new_token in text_streamer:
        if new_token != "<":
            partial_message += new_token
            yield partial_message


gr.ChatInterface(predict).launch(share=True, debug=True)


# * | // MARK: bak, unsloth template =====================================
# unsloth_template = \
#     "{{ bos_token }}"\
#     "{{ 'You are a helpful assistant to the user\n' }}"\
#     "{% for message in messages %}"\
#         "{% if message['role'] == 'user' %}"\
#             "{{ '>>> User: ' + message['content'] + '\n' }}"\
#         "{% elif message['role'] == 'assistant' %}"\
#             "{{ '>>> Assistant: ' + message['content'] + eos_token + '\n' }}"\
#         "{% endif %}"\
#     "{% endfor %}"\
#     "{% if add_generation_prompt %}"\
#         "{{ '>>> Assistant: ' }}"\
#     "{% endif %}"
# unsloth_eos_token = "eos_token"
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = (unsloth_template, unsloth_eos_token,), # You must provide a template and EOS token
#     map_eos_token = True, # Maps <|im_end|> to </s> instead
# )
