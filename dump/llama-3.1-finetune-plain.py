# -*- coding: utf-8 -*-
# pip uninstall -y unsloth xformers trl peft accelerate bitsandbytes datasets transformers gradio ipdb torch wandb
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
# pip install torch==2.3.0 gradio ipdb wandb
from datasets import load_dataset
from peft import LoraConfig
from transformers import  AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging
from trl import SFTTrainer
import os
import torch

# 현재 사용 중인 GPU의 주요 아키텍처 버전을 반환 8버전 이상 시 bfloat16 활용
if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

# BitsAndBytesConfig 객체활용 양자화 설정
# 모델을 4비트 양자화하여 로드
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map={"": 0}
    # device_map="auto"
)
model.config.use_cache = True
model.config.pretraining_tp = 1


tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = load_dataset("NoNameFactory/callcenter", '금융보험_상품 가입 및 해지', split = "train")

# ShareGPT {"from": "human", "value" : "Hi"} to ChatML {"role": "user", "content" : "Hi"}
def convert_chat_format(chat):
    conversion_map = {"human": "user", "gpt": "assistant"}
    return [{"role": conversion_map.get(entry["from"], entry["from"]), "content": entry["value"]} for entry in chat]

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [ convert_chat_format(convo) for convo in convos]
    return { "ChatML" : texts, }
pass

dataset = dataset.map(formatting_prompts_func, batched = True,)

# ChatML {"role": "user", "content" : "Hi"} to Llama3 <|begin_of_text|><|start_header_id|>user<|end_header_id|>Hello!<|eot_id|>
def formatting_prompts_func2(examples):
    convos = examples["ChatML"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

dataset = dataset.map(formatting_prompts_func2, batched = True,)

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

dataset = dataset.select(range(100))

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=128, # None for unlimited
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)
trainer.train()

messages = [
    {"role": "system", "content": "친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘."},
    {"role": "user", "content": "대한민국에서 가장 가볼만한 곳은 어디니?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=True,
    temperature=1,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))