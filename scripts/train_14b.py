#!/usr/bin/env python3
"""
TokioAI 14B v2 Training Script
QLoRA 4-bit fine-tuning on a single T4 GPU (16 GB VRAM).

This model is too large for bf16 on T4, so we use:
- 4-bit quantization (nf4) to load the model in ~8 GB
- Paged AdamW 8-bit optimizer to reduce optimizer state
- Lower LoRA rank (8 vs 16) to save VRAM
- Gradient checkpointing to trade compute for memory

Usage:
    python3 train_14b.py

Output:
    - LoRA adapter: tokioai-14b-v2/
    (Merge and GGUF conversion done separately -- needs more RAM)
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

# Helps PyTorch reuse freed GPU memory blocks
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
DATASET_PATH = "tokioai_dataset_v2.jsonl"
OUTPUT_DIR = "tokioai-14b-v2"

# Hyperparameters (more conservative than 3B)
EPOCHS = 4
BATCH_SIZE = 1
GRAD_ACCUM = 8       # Larger accumulation = smoother gradients
LR = 5e-5            # Lower LR for larger model
MAX_LENGTH = 1536    # Reduced from 2048 to save VRAM
LORA_RANK = 8        # Lower rank to save VRAM
LORA_ALPHA = 16

# ============================================================
# STEP 1: Load and format dataset
# ============================================================
print(f"[{datetime.now()}] === TokioAI 14B v2 Training (QLoRA) ===")
print(f"GPU: {torch.cuda.get_device_name(0)}")

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
# STEP 2: Configure 4-bit quantization
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Load weights as 4-bit integers
    bnb_4bit_quant_type="nf4",            # NormalFloat4: optimal for normally
                                          # distributed weights (which LLMs have)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16 during forward pass
    bnb_4bit_use_double_quant=True,       # Quantize the quantization constants
                                          # (saves ~0.4 bits/param extra)
)

# ============================================================
# STEP 3: Load model in 4-bit
# ============================================================
print(f"[{datetime.now()}] Loading 14B model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

# IMPORTANT: We skip prepare_model_for_kbit_training() here.
# That function casts layer norms to float32, which adds ~2 GB VRAM
# and causes OOM on T4 with 14B. Instead, we do the essentials manually:
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
for name, param in model.named_parameters():
    param.requires_grad = False  # Freeze all base params
# LoRA's get_peft_model will enable gradients on adapter params

# ============================================================
# STEP 4: Configure and apply LoRA
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
# STEP 5: Configure training
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
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=0.3,             # Aggressive clipping (stability for large model)
    optim="paged_adamw_8bit",      # 8-bit optimizer saves ~2 GB VRAM
                                   # "Paged" = offloads to CPU if GPU runs out
    report_to="none",
    max_length=MAX_LENGTH,
    packing=False,
    dataset_text_field="text",
    loss_type="nll",               # Fix: avoids TRL 1.14 chunked CE bug
                                   # (functools.partial serialization error)
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
print(f"[{datetime.now()}] DONE -- merge and GGUF conversion needed separately")
