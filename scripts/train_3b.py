#!/usr/bin/env python3
"""
TokioAI 3B v2 Training Script
Full precision LoRA fine-tuning on a single T4 GPU (16 GB VRAM).

Usage:
    python3 train_3b.py

Requirements:
    pip install torch transformers datasets peft trl bitsandbytes accelerate

Output:
    - LoRA adapter:  tokioai-3b-v2/
    - Merged model:  tokioai-3b-v2-merged/
    - GGUF F16:      tokioai-3b-v2-gguf/tokioai-3b-v2-f16.gguf
    - GGUF Q4_K_M:   tokioai-3b-v2-gguf/tokioai-3b-v2-Q4_K_M.gguf
    - Ollama model:  tokioai-3b-v2 (registered in ollama)
"""
import os
import json
import time
import subprocess
import torch
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

# ============================================================
# CONFIGURATION -- Adjust these for your setup
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = "tokioai_dataset_v2.jsonl"
OUTPUT_DIR = "tokioai-3b-v2"
MERGED_DIR = "tokioai-3b-v2-merged"
GGUF_DIR = "tokioai-3b-v2-gguf"

# Training hyperparameters
EPOCHS = 8           # More epochs for small model (it can memorize)
BATCH_SIZE = 1       # Per-device batch size (keep at 1 for VRAM)
GRAD_ACCUM = 4       # Effective batch size = BATCH_SIZE * GRAD_ACCUM = 4
LR = 1e-4            # Learning rate (aggressive for small model)
MAX_LENGTH = 2048    # Max sequence length in tokens
LORA_RANK = 16       # LoRA rank (higher = more capacity, more VRAM)
LORA_ALPHA = 32      # LoRA alpha (rule of thumb: 2 * rank)

# Post-training
DO_MERGE = True      # Merge adapter into base model
DO_GGUF = True       # Convert to GGUF format
DO_QUANTIZE = True   # Quantize to Q4_K_M
DO_OLLAMA = True     # Create Ollama model

# ============================================================
# STEP 1: Load and format dataset
# ============================================================
print(f"[{datetime.now()}] === TokioAI 3B v2 Training ===")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

data = []
with open(DATASET_PATH) as f:
    for line in f:
        example = json.loads(line.strip())
        # apply_chat_template converts messages -> model-specific token format
        # This is CRITICAL: never format manually, always use the tokenizer
        formatted_text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        data.append({"text": formatted_text})

print(f"Dataset: {len(data)} examples")
dataset = Dataset.from_list(data)

# ============================================================
# STEP 2: Load base model (full bf16 -- no quantization for 3B)
# ============================================================
print(f"[{datetime.now()}] Loading model (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",           # Auto-place on GPU
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # bf16 is more stable than fp16
)

# ============================================================
# STEP 3: Configure and apply LoRA
# ============================================================
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",       # MLP layers
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# ============================================================
# STEP 4: Configure training
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
    gradient_checkpointing=True,  # Saves ~40% VRAM, costs ~20% compute
    max_grad_norm=1.0,
    optim="adamw_torch",
    report_to="none",
    max_length=MAX_LENGTH,
    packing=False,
    dataset_text_field="text",
)

# ============================================================
# STEP 5: Train
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

# ============================================================
# STEP 6: Merge adapter with base model
# ============================================================
if DO_MERGE:
    print(f"[{datetime.now()}] Merging adapter with base model...")
    del model
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # CPU for merge (avoids GPU OOM)
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    merged = merged.merge_and_unload()  # W_merged = W_base + A * B
    merged.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Merged model saved to {MERGED_DIR}/")
    del merged, base_model

# ============================================================
# STEP 7: Convert to GGUF
# ============================================================
if DO_GGUF:
    print(f"[{datetime.now()}] Converting to GGUF...")
    os.makedirs(GGUF_DIR, exist_ok=True)

    llama_cpp_dir = os.path.expanduser("~/llama.cpp")
    if not os.path.exists(llama_cpp_dir):
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp", llama_cpp_dir],
            timeout=120,
        )
        subprocess.run(
            ["pip", "install", "-r", f"{llama_cpp_dir}/requirements.txt"],
            capture_output=True, timeout=120,
        )

    f16_path = os.path.join(GGUF_DIR, "tokioai-3b-v2-f16.gguf")
    subprocess.run(
        [
            "python3", f"{llama_cpp_dir}/convert_hf_to_gguf.py",
            MERGED_DIR, "--outfile", f16_path, "--outtype", "f16",
        ],
        timeout=600,
    )
    print(f"GGUF F16: {os.path.getsize(f16_path) / 1e9:.1f} GB")

    # ============================================================
    # STEP 8: Quantize to Q4_K_M
    # ============================================================
    if DO_QUANTIZE:
        print(f"[{datetime.now()}] Quantizing to Q4_K_M...")
        subprocess.run(
            ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=llama_cpp_dir, timeout=60, capture_output=True,
        )
        subprocess.run(
            ["cmake", "--build", "build", "--config", "Release", "-j4",
             "--target", "llama-quantize"],
            cwd=llama_cpp_dir, timeout=300, capture_output=True,
        )

        quantize_bin = f"{llama_cpp_dir}/build/bin/llama-quantize"
        q4_path = os.path.join(GGUF_DIR, "tokioai-3b-v2-Q4_K_M.gguf")

        if os.path.exists(quantize_bin):
            subprocess.run([quantize_bin, f16_path, q4_path, "Q4_K_M"], timeout=300)
            print(f"Q4_K_M: {os.path.getsize(q4_path) / 1e9:.1f} GB")

# ============================================================
# STEP 9: Create Ollama model
# ============================================================
if DO_OLLAMA:
    print(f"[{datetime.now()}] Creating Ollama model...")
    q4_path = os.path.join(GGUF_DIR, "tokioai-3b-v2-Q4_K_M.gguf")
    modelfile_path = "Modelfile-3b-v2"

    with open(modelfile_path, "w") as f:
        f.write(f"FROM {os.path.abspath(q4_path)}\n")
        f.write("PARAMETER temperature 0.1\n")
        f.write("PARAMETER top_p 0.9\n")
        f.write('PARAMETER stop "<|im_end|>"\n')
        f.write('PARAMETER stop "</tool_call>"\n')
        f.write("PARAMETER num_ctx 4096\n")

    subprocess.run(["ollama", "create", "tokioai-3b-v2", "-f", modelfile_path], timeout=120)

print(f"[{datetime.now()}] === DONE ===")
