#!/usr/bin/env python3
"""
TokioAI 72B v2 Training Script
QLoRA 4-bit fine-tuning on 2x L4 GPUs (48 GB total VRAM).

The 72B model in 4-bit needs ~40 GB VRAM. device_map="auto" splits
the model across both GPUs automatically.

Usage:
    python3 train_72b.py

Hardware:
    - 2x NVIDIA L4 (24 GB each)
    - 96 GB system RAM
    - 500 GB disk (model shards + checkpoints)

Output:
    - LoRA adapter: tokioai-72b-v2/
    (Merge needs ~150 GB RAM -- do on a high-memory machine)
"""
import os
import json
import time
import torch
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
DATASET_PATH = "tokioai_dataset_v2.jsonl"
OUTPUT_DIR = "tokioai-72b-v2"

# Most conservative hyperparameters
EPOCHS = 3            # Large models learn fast
BATCH_SIZE = 1
GRAD_ACCUM = 16       # Very large accumulation for stability
LR = 2e-5             # Very low LR (large model is sensitive)
MAX_LENGTH = 1024     # CRITICAL: 2048 caused OOM at step 2 on 2xL4
                      # 1024 gives ~600 MB headroom
LORA_RANK = 8
LORA_ALPHA = 16

# ============================================================
# STEP 1: Dataset
# ============================================================
print(f"[{datetime.now()}] === TokioAI 72B v2 Training (QLoRA 2xGPU) ===")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_mem / 1e9:.1f} GB)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

data = []
with open(DATASET_PATH) as f:
    for line in f:
        example = json.loads(line.strip())
        formatted_text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        data.append({"text": formatted_text})
print(f"Dataset: {len(data)} examples")
dataset = Dataset.from_list(data)

# ============================================================
# STEP 2: 4-bit quantization (same config for all QLoRA models)
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ============================================================
# STEP 3: Load 72B model (~10 min to download + load 37 shards)
# ============================================================
print(f"[{datetime.now()}] Loading 72B model (this takes ~10 min)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",           # Splits across GPU 0 and GPU 1
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
for name, param in model.named_parameters():
    param.requires_grad = False

# ============================================================
# STEP 4: LoRA
# ============================================================
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# ============================================================
# STEP 5: Training config
# ============================================================
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=5,               # Log more frequently (fewer total steps)
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=0.3,
    optim="paged_adamw_8bit",
    report_to="none",
    max_length=MAX_LENGTH,
    packing=False,
    dataset_text_field="text",
    loss_type="nll",
)

# ============================================================
# STEP 6: Train
# ============================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    processing_class=tokenizer,
)

print(f"[{datetime.now()}] Starting training...")
start = time.time()
trainer.train()
elapsed = time.time() - start
print(f"[{datetime.now()}] Training done in {elapsed / 60:.1f} min")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}/")
print(f"[{datetime.now()}] DONE")
print("NOTE: Merge requires ~150 GB RAM (72B in bf16). Export adapter and merge on a high-memory machine.")
