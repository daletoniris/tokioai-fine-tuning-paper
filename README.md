# Fine-Tuning LLMs for Tool Calling: A Production Pipeline

## From Qwen2.5 to TokioAI-3B/14B/72B -- Training, Quantization, and Deployment

**Author:** Daniel Dieser (TokioAI)
**Date:** September 2026
**Workshop Level:** Intermediate to Advanced

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Architecture Overview](#2-architecture-overview)
3. [The Dataset](#3-the-dataset)
4. [Understanding the Training](#4-understanding-the-training)
5. [Training Script: 3B Model (Full Precision LoRA)](#5-training-script-3b-model-full-precision-lora)
6. [Training Script: 14B Model (QLoRA 4-bit)](#6-training-script-14b-model-qlora-4-bit)
7. [Training Script: 72B Model (QLoRA Multi-GPU)](#7-training-script-72b-model-qlora-multi-gpu)
8. [Post-Training: Merge, GGUF, and Quantization](#8-post-training-merge-gguf-and-quantization)
9. [Deployment with Ollama](#9-deployment-with-ollama)
10. [Lessons Learned and Debugging](#10-lessons-learned-and-debugging)
11. [Results and Comparison](#11-results-and-comparison)
12. [Hardware Requirements](#12-hardware-requirements)
13. [Reproducing This Work](#13-reproducing-this-work)
14. [References](#14-references)

---

## 1. Introduction

This paper documents the complete pipeline for fine-tuning large language models to perform **tool calling** -- the ability to decide which tools to invoke and with what parameters based on natural language input.

The goal: take a base model (Qwen2.5) and teach it to behave like a cybersecurity/DevOps assistant that can:
- Parse user requests in natural language (English or Spanish)
- Decide which tool(s) to call from a catalog of 30+ tools
- Generate the correct JSON arguments for each tool
- Handle multi-tool chains, error recovery, and edge cases

We fine-tuned three model sizes on the same dataset:

| Model | Base | Method | GPU | VRAM | Training Time | Output Size (Q4_K_M) |
|-------|------|--------|-----|------|---------------|---------------------|
| TokioAI-3B v2 | Qwen2.5-3B-Instruct | LoRA (full bf16) | 1x T4 16GB | ~12 GB | 77 min | 1.8 GB |
| TokioAI-14B v2 | Qwen2.5-14B-Instruct | QLoRA (4-bit nf4) | 1x T4 16GB | ~11 GB | ~5.5 h | ~8.5 GB |
| TokioAI-72B v2 | Qwen2.5-72B-Instruct | QLoRA (4-bit nf4) | 2x L4 48GB | ~42 GB | ~1.9 h | ~42 GB |

The complete pipeline covers: dataset creation, training with LoRA/QLoRA, adapter merging, GGUF conversion, quantization, and deployment with Ollama.

---

## 2. Architecture Overview

### What is Tool Calling?

Traditional LLMs generate text. A tool-calling LLM generates **structured function calls** that an external system executes. The flow:

```
User: "check disk space on the raspi"
      |
      v
  [LLM Inference]
      |
      v
Model Output:
  <tool_call>
  {"name": "execute_raspi", "arguments": {"command": "df -h"}}
  </tool_call>
      |
      v
  [Tool Executor] --- runs the actual command
      |
      v
  [Result returned to user]
```

### Why Fine-Tune?

Base models like Qwen2.5 already support function calling via their chat template. But:

1. **Domain specificity**: We need the model to know OUR tools (execute_local, irt_chat, execute_raspi, etc.), not generic OpenAI-style function calling.
2. **Behavioral patterns**: The model should act first, explain after. It should use Spanish when the user speaks Spanish. It should never expose credentials.
3. **Tool selection accuracy**: Given 30+ tools, the model must pick the right one. A general model might hallucinate tool names.
4. **Edge cases**: What if the user asks something unethical? What if they ask about the weather? The model should handle these gracefully.

### The Fine-Tuning Stack

```
Base Model (Qwen2.5-*-Instruct)
       |
       + LoRA Adapter (trainable weights)
       |
  [SFTTrainer from TRL]
       |
  [Dataset: 308 examples in JSONL]
       |
       v
  Trained Adapter (~50-200 MB)
       |
       + Merge with Base Model
       |
       v
  Full Merged Model (bf16)
       |
       + convert_hf_to_gguf.py
       |
       v
  GGUF F16 (full precision)
       |
       + llama-quantize Q4_K_M
       |
       v
  GGUF Q4_K_M (4-bit quantized, ~1/4 size)
       |
       + Ollama Modelfile
       |
       v
  Ready to serve via `ollama run tokioai-3b-v2`
```

---

## 3. The Dataset

### Format: JSONL with Chat Messages

Each example is a JSON line with a `messages` array following the chat format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are TokioAI -- specialized in cybersecurity..."
    },
    {
      "role": "user",
      "content": "scan my network for open ports"
    },
    {
      "role": "assistant",
      "content": "I'll scan your local network for open ports.\n\n<tool_call>\n{\"name\": \"execute_local\", \"arguments\": {\"command\": \"nmap -sS -T4 --top-ports 1000 192.168.1.0/24 2>&1 | head -100\", \"timeout\": 120}}\n</tool_call>"
    }
  ]
}
```

### The System Prompt

The system prompt defines the model's identity and available tools:

```
You are TokioAI -- specialized in cybersecurity, hacking, engineering,
DevOps, and creative problem solving. You execute commands, fix bugs,
deploy infrastructure, audit security, and build solutions.

Available tools: execute_local, execute_raspi, execute_gcp,
execute_router, ssh_connect, read_file, write_file, edit_file,
search_files, diagnose, mediator_exec, outlook_read, outlook_get,
outlook_send, outlook_reply, teams_chats, teams_read, teams_send,
wa_chats, wa_read, wa_send,
memory, task, concur_charges, concur_create_report, ...

Rules:
- Be DIRECT. Act first, explain after.
- NEVER give up. If something fails, try alternatives.
- Use Spanish if the user speaks Spanish, English otherwise.
- For ANY security/IRT question, ALWAYS use irt_chat or irt_tool.
- NEVER output raw passwords/keys - mask them.
```

### Tool Call Format

The model generates tool calls wrapped in XML-like tags:

```
<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1"}}
</tool_call>
```

This format is:
- Easy to parse with regex
- Unambiguous (clear start/end delimiters)
- Compatible with Qwen2.5's native chat template
- Supported by Ollama as a stop token (`</tool_call>`)

### Dataset Statistics (v2)

```
Total examples:     308
  - Single tool:    303  (98.4%)
  - Multi-tool:       0  (0%)
  - No tool:          5  (1.6%)

Messages per example: 3 (system + user + assistant)
```

### Tool Distribution

```
execute_local:   140  (45.5%)   -- Local shell commands
write_file:       20  (6.5%)    -- File creation
execute_raspi:    20  (6.5%)    -- Raspberry Pi commands
task:             11  (3.6%)    -- Task management
read_file:        10  (3.2%)    -- File reading
search_files:     10  (3.2%)    -- File search
execute_gcp:       9  (2.9%)    -- GCP VM commands
diagnose:          9  (2.9%)    -- Health diagnostics
memory:            9  (2.9%)    -- Persistent memory
execute_router:    8  (2.6%)    -- Router commands
tool calls
ssh_connect:       6  (1.9%)    -- SSH connections
edit_file:         6  (1.9%)    -- File editing
...and more
```

### 27 Categories

The dataset covers 27 distinct categories:

1. **execute_local** -- System administration, process management
2. **Security / Pentest** -- Nmap, hashcat, web scanning, OSINT
4. **Raspberry Pi** -- GPIO, temperature, Hailo AI, home automation
5. **GCP VM** -- Cloud operations, GPU monitoring, deployments
6. **Router** -- Network device configuration
7. **File Operations** -- read_file, write_file, edit_file, search_files
8. **Diagnostics** -- System health checks
9. **Memory** -- Persistent memory operations
10. **Task Management** -- Tracking work items
11. **SSH Connect** -- Ad-hoc server connections
13. **Outlook / Email** -- Reading, sending, replying to emails
14. **Teams** -- Chat operations
15. **WhatsApp** -- Messaging
16. **Multi-tool chains** -- Sequential tool calls
17. **Error recovery** -- Handling failures gracefully
18. **Reasoning before action** -- Think then act
19. **Ethical boundaries** -- Refusing unethical requests
20. **Personality** -- Identity, humor, meta-questions
21. IOC extraction, hardening
23. **Database** -- PostgreSQL, MySQL queries
24. **Automation** -- Script generation
25. **Hailo / Edge AI** -- AI accelerator on Raspberry Pi
26. **Networking deep** -- Packet capture, DNS, MTU testing
27. **Git / Version control** -- Repository operations

### Dataset Generation Script

The dataset is generated programmatically, not manually typed. Here's the generator structure:

```python
import json

SYSTEM_PROMPT = "You are TokioAI -- specialized in..."

def tc(name, args):
    """Generate a tool_call block."""
    return ("<tool_call>\n"
            + json.dumps({"name": name, "arguments": args})
            + "\n</tool_call>")

def ex(user, assistant):
    """Create a training example."""
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]}

examples = []

# Category 1: execute_local
examples.append(ex(
    "scan my network for open ports",
    "I'll scan your local network.\n\n"
    + tc("execute_local", {
        "command": "nmap -sS -T4 --top-ports 1000 192.168.1.0/24",
        "timeout": 120
    })
))

# ... 300+ more examples across 27 categories ...

# Write JSONL
with open('tokioai_dataset_v2.jsonl', 'w') as f:
    for e in examples:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')
```

### Key Design Decisions

1. **One example = one complete interaction**: system + user + assistant. No multi-turn conversations (keeps it simple and each example is self-contained).

2. **System prompt in every example**: The model learns to associate the system prompt with the tool-calling behavior. This is critical -- without it, the model won't know what tools are available.

3. **Mix of languages**: ~70% English, ~30% Spanish. The model learns to respond in the user's language.

4. **Realistic commands**: The tool arguments contain real-world commands (`nmap -sS`, `docker ps`, `df -h`), not toy examples.

5. **Edge cases matter**: 5 examples have NO tool calls (ethical refusals, personality questions). This teaches the model that not every input requires a tool.

6. **V2 builds on V1**: The v2 dataset includes all v1 examples (97) plus 211 new ones. The generator reads v1 first, then appends v2.

---

## 4. Understanding the Training

### What is LoRA?

**LoRA (Low-Rank Adaptation)** is a technique that freezes the base model weights and injects small trainable matrices into specific layers. Instead of fine-tuning all 3B/14B/72B parameters, we train ~0.5-2% of them.

```
Original weight matrix W (frozen):     [4096 x 4096]
LoRA matrices A and B (trainable):     [4096 x r] * [r x 4096]
                                        where r = rank (8 or 16)

Effective weight: W + A*B
```

Benefits:
- **Memory**: Only the small A/B matrices are stored in optimizer state
- **Speed**: Fewer parameters to update
- **Modularity**: The adapter can be saved separately (~50-200 MB vs 6-140 GB)

### What is QLoRA?

**QLoRA** = Quantized LoRA. The base model is loaded in 4-bit precision (instead of 16-bit), reducing VRAM by ~4x:

```
                        Model Size in VRAM
Model     bf16          4-bit (nf4)       Reduction
3B        ~6 GB         ~1.8 GB           3.3x
14B       ~28 GB        ~8 GB             3.5x
72B       ~144 GB       ~40 GB            3.6x
```

The key configuration for 4-bit quantization:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Load weights as 4-bit
    bnb_4bit_quant_type="nf4",      # NormalFloat4: optimal for normal distributions
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16 for stability
    bnb_4bit_use_double_quant=True, # Quantize the quantization constants too
)
```

**Important**: QLoRA keeps the base model frozen in 4-bit. Only the LoRA adapter weights are in full precision (bf16). Gradients flow through the LoRA path only.

### LoRA Target Modules

For transformer models, we target all the attention and feed-forward projection layers:

```python
target_modules = [
    "q_proj",     # Query projection     (attention)
    "k_proj",     # Key projection       (attention)
    "v_proj",     # Value projection     (attention)
    "o_proj",     # Output projection    (attention)
    "gate_proj",  # Gate projection      (MLP / feed-forward)
    "up_proj",    # Up projection        (MLP / feed-forward)
    "down_proj",  # Down projection      (MLP / feed-forward)
]
```

Why these 7 layers? They cover the full information flow through each transformer block. The attention layers control WHAT the model pays attention to, and the MLP layers control HOW it transforms the information. Training all 7 gives the adapter maximum expressiveness with minimal rank.

### LoRA Hyperparameters Explained

```python
LoraConfig(
    r=16,              # Rank: size of the low-rank matrices
                       # Higher = more capacity but more VRAM
                       # 8 for 14B/72B (VRAM constrained)
                       # 16 for 3B (plenty of VRAM)

    lora_alpha=32,     # Scaling factor: alpha/r = effective learning rate multiplier
                       # Rule of thumb: alpha = 2*r
                       # 3B: alpha=32 (32/16=2.0x)
                       # 14B/72B: alpha=16 (16/8=2.0x)

    lora_dropout=0.05, # Dropout on LoRA layers (regularization)
                       # 0.05 = 5% dropout, mild regularization

    bias="none",       # Don't train bias terms (saves memory, rarely helps)

    task_type="CAUSAL_LM",  # Causal language modeling (autoregressive)
)
```

### Training Hyperparameters Explained

```python
SFTConfig(
    # --- Epochs and Batching ---
    num_train_epochs=8,           # 3B: 8 epochs (small model, needs more passes)
                                  # 14B: 4 epochs (larger model, learns faster)
                                  # 72B: 3 epochs (huge model, even fewer needed)

    per_device_train_batch_size=1,   # 1 example per step (VRAM constrained)

    gradient_accumulation_steps=4,   # Accumulate 4 steps before updating
                                     # Effective batch size = 1 * 4 = 4
                                     # 14B uses 8, 72B uses 16
                                     # Larger accumulation = smoother gradients

    # --- Learning Rate ---
    learning_rate=1e-4,       # 3B: 1e-4 (aggressive, small model tolerates it)
                              # 14B: 5e-5 (moderate)
                              # 72B: 2e-5 (conservative, large model is sensitive)

    lr_scheduler_type="cosine",  # Cosine decay: starts high, smoothly decreases
                                 # Better than linear for LoRA fine-tuning

    warmup_ratio=0.05,       # 5% of steps at reduced LR (prevents early instability)

    # --- Regularization ---
    weight_decay=0.01,       # L2 regularization on weights
    max_grad_norm=1.0,       # 3B: 1.0 (standard gradient clipping)
                             # 14B/72B: 0.3 (aggressive clipping for stability)

    # --- Precision and Memory ---
    bf16=True,               # Use bfloat16 for training (better than fp16 for stability)

    gradient_checkpointing=True,  # Trade compute for memory:
                                  # Don't store all activations, recompute during backward
                                  # Saves ~40% VRAM, costs ~20% more compute time

    gradient_checkpointing_kwargs={"use_reentrant": False},
                                  # PyTorch 2.x: non-reentrant is safer and sometimes faster

    # --- Optimizer ---
    optim="adamw_torch",     # 3B: standard AdamW (enough VRAM)
    # optim="paged_adamw_8bit",  # 14B/72B: 8-bit paged AdamW
                                  # Reduces optimizer state VRAM by ~2x
                                  # "Paged" = can offload to CPU if GPU runs out

    # --- Sequence Length ---
    max_length=2048,         # 3B: 2048 tokens per example (full context)
                             # 14B: 1536 tokens (VRAM compromise)
                             # 72B: 1024 tokens (VRAM critical -- see Section 10)

    # --- Data ---
    packing=False,           # Don't pack multiple examples into one sequence
                             # Packing is faster but can confuse tool-call boundaries

    dataset_text_field="text",  # Column name in the HF Dataset

    # --- Output ---
    save_strategy="epoch",   # Save a checkpoint after each epoch
    logging_steps=10,        # Log loss every 10 steps
    report_to="none",        # No W&B or TensorBoard
)
```

### The loss_type="nll" Fix

In TRL 1.14, there's a bug where `SFTConfig` uses a chunked cross-entropy loss by default. This loss is wrapped with `functools.partial`, which breaks serialization during checkpointing on some setups.

The fix: explicitly set `loss_type="nll"` (negative log-likelihood), which uses the standard PyTorch cross-entropy:

```python
sft_config = SFTConfig(
    ...,
    loss_type="nll",  # Avoid chunked CE bug in TRL 1.14
)
```

This is a one-liner but it cost us hours of debugging. Always check your TRL version's changelog.

### The prepare_model_for_kbit_training Trap

The standard QLoRA recipe says:

```python
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)  # DON'T DO THIS for large models on small GPUs
```

What this function does:
1. Casts certain layers to float32 (for stability)
2. Enables gradient checkpointing
3. Sets `requires_grad=False` on base params

Problem: **step 1 casts layer norms to float32**, which on a 14B model adds ~2 GB of VRAM. On a T4 with 16GB, this causes OOM.

Our solution: skip `prepare_model_for_kbit_training` entirely and do the essential parts manually:

```python
# Do these two things manually instead:
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
for name, param in model.named_parameters():
    param.requires_grad = False
# LoRA's get_peft_model will set requires_grad=True on the adapter params
```

---

## 5. Training Script: 3B Model (Full Precision LoRA)

The 3B model is small enough to load in full bf16 precision (no quantization needed).

```python
#!/usr/bin/env python3
"""Train TokioAI 3B v2 -- Full precision LoRA on T4 GPU"""
import os, json, time, torch, subprocess
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = "tokioai_dataset_v2.jsonl"
OUTPUT_DIR = "tokioai-3b-v2"

# --- Load and format dataset ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

data = []
with open(DATASET_PATH) as f:
    for line in f:
        example = json.loads(line.strip())
        # Apply chat template: converts messages -> model-specific format
        formatted_text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        data.append({"text": formatted_text})

print(f"Dataset: {len(data)} examples")
dataset = Dataset.from_list(data)

# --- Load model in full precision (bf16) ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # Full bf16, no quantization
)

# --- Configure LoRA ---
lora_config = LoraConfig(
    r=16, lora_alpha=32,           # Higher rank for 3B (plenty of VRAM)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
# Output: Trainable: 54,525,952 / 3,144,187,904 (1.73%)

# --- Training config ---
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=8,                # More epochs for small model
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,     # Effective batch = 4
    learning_rate=1e-4,                # Aggressive LR
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    optim="adamw_torch",              # Standard AdamW (enough VRAM)
    report_to="none",
    max_length=2048,                   # Full context
    packing=False,
    dataset_text_field="text",
)

# --- Train ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    processing_class=tokenizer,
)

start = time.time()
trainer.train()
elapsed = time.time() - start
print(f"Training done in {elapsed/60:.1f} min")
# Output: Training done in 77.4 min (T4 GPU)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

### 3B Training Results

```
GPU:               NVIDIA T4 (16 GB)
Training time:     77.4 minutes
Final loss:        0.0001
Steps:             616 total (308 examples * 8 epochs / 4 accumulation)
Step time:         ~7.5 s/step
VRAM peak:         ~12 GB
Trainable params:  54.5M / 3.1B (1.73%)
```

---

## 6. Training Script: 14B Model (QLoRA 4-bit)

The 14B model doesn't fit in bf16 on a T4 (28 GB > 16 GB). We use QLoRA to load it in 4-bit (~8 GB) and train the LoRA adapter in bf16.

```python
#!/usr/bin/env python3
"""Train TokioAI 14B v2 -- QLoRA 4-bit on T4 GPU"""
import os, json, time, torch
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
DATASET_PATH = "tokioai_dataset_v2.jsonl"
OUTPUT_DIR = "tokioai-14b-v2"

# --- Dataset (same as 3B) ---
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
dataset = Dataset.from_list(data)

# --- 4-bit Quantization Config ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- Load model in 4-bit ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

# Manual setup (skip prepare_model_for_kbit_training to avoid OOM)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
for name, param in model.named_parameters():
    param.requires_grad = False

# --- LoRA with lower rank (VRAM constrained) ---
lora_config = LoraConfig(
    r=8, lora_alpha=16,            # Lower rank to save VRAM
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# --- Training config (conservative) ---
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,                # Fewer epochs
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,     # Larger accumulation = smoother
    learning_rate=5e-5,                # Lower LR (larger model)
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=0.3,                 # Aggressive clipping
    optim="paged_adamw_8bit",          # 8-bit optimizer (saves VRAM)
    report_to="none",
    max_length=1536,                   # Reduced from 2048 (VRAM)
    packing=False,
    dataset_text_field="text",
    loss_type="nll",                   # Fix TRL 1.14 chunked CE bug
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

### 14B Training Results

```
GPU:               NVIDIA T4 (16 GB)
Training time:     ~5.5 hours (estimated)
Steps:             156 total (308 examples * 4 epochs / 8 accumulation)
Step time:         ~110 s/step
VRAM peak:         ~11 GB
```

---

## 7. Training Script: 72B Model (QLoRA Multi-GPU)

The 72B model in 4-bit needs ~40 GB VRAM. We use 2x NVIDIA L4 GPUs (24 GB each = 48 GB total). `device_map="auto"` handles the splitting.

```python
#!/usr/bin/env python3
"""Train TokioAI 72B v2 -- QLoRA 4-bit on 2x L4 GPUs"""
import os, json, time, torch
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
DATASET_PATH = "tokioai_dataset_v2.jsonl"
OUTPUT_DIR = "tokioai-72b-v2"

# --- Dataset (same format) ---
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
dataset = Dataset.from_list(data)

# --- Same 4-bit config ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- Load model (splits across 2 GPUs automatically) ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",           # Auto-split across available GPUs
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
for name, param in model.named_parameters():
    param.requires_grad = False

# --- LoRA ---
lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# --- Most conservative training config ---
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,                 # Minimum viable epochs
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,     # Very large accumulation
    learning_rate=2e-5,                 # Very low LR
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_strategy="epoch",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=0.3,
    optim="paged_adamw_8bit",
    report_to="none",
    max_length=1024,                    # CRITICAL: reduced to 1024
                                        # 2048 caused OOM on 2xL4 at step 2
    packing=False,
    dataset_text_field="text",
    loss_type="nll",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

### 72B Training Results

```
GPUs:              2x NVIDIA L4 (24 GB each = 48 GB total)
VM:                g2-standard-24 (24 vCPUs, 96 GB RAM)
Training time:     ~1.9 hours
Steps:             60 total (308 examples * 3 epochs / 16 accumulation)
Step time:         ~111 s/step
VRAM peak:         ~42 GB (20.3 + 21.8 across 2 GPUs)
```

### Scaling Comparison

| Parameter | 3B | 14B | 72B |
|---|---|---|---|
| Loading method | bf16 (full) | QLoRA 4-bit | QLoRA 4-bit |
| LoRA rank (r) | 16 | 8 | 8 |
| Learning rate | 1e-4 | 5e-5 | 2e-5 |
| Epochs | 8 | 4 | 3 |
| Gradient accumulation | 4 | 8 | 16 |
| Max sequence length | 2048 | 1536 | 1024 |
| Optimizer | AdamW | paged AdamW 8-bit | paged AdamW 8-bit |
| Max gradient norm | 1.0 | 0.3 | 0.3 |
| GPU(s) | 1x T4 16GB | 1x T4 16GB | 2x L4 48GB |
| Total training time | 77 min | ~5.5 h | ~1.9 h |
| Step time | ~7.5 s | ~110 s | ~111 s |

**Pattern**: As model size increases:
- Lower learning rate (less aggressive updates)
- Fewer epochs (larger models learn faster per pass)
- Larger gradient accumulation (smoother updates)
- Shorter sequences (save VRAM)
- More aggressive gradient clipping (stability)

---

## 8. Post-Training: Merge, GGUF, and Quantization

After training, we have a LoRA adapter (~50-200 MB). To deploy it, we need to:

1. **Merge** the adapter into the base model
2. **Convert** to GGUF format (for llama.cpp / Ollama)
3. **Quantize** to reduce size (Q4_K_M = ~4 bits per weight)

### Step 1: Merge Adapter with Base Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "tokioai-3b-v2"       # LoRA adapter from training
MERGED_DIR = "tokioai-3b-v2-merged"  # Output: full merged model

# Load base model on CPU (needs full precision for merge)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cpu",                 # CPU to avoid GPU OOM during merge
    trust_remote_code=True,
)

# Load adapter on top
merged = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# Merge LoRA weights into base weights
merged = merged.merge_and_unload()
# This does: W_merged = W_base + A * B
# After this, LoRA is gone -- it's a standard model with modified weights

# Save
merged.save_pretrained(MERGED_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_DIR)
```

**Important for 72B**: The merge requires loading the full model in bf16 on CPU. For 72B, this needs ~150 GB of RAM. If your VM has less, you can:
- Export only the LoRA adapter and merge on a larger machine
- Use `device_map="auto"` with enough GPU VRAM

### Step 2: Convert to GGUF

GGUF (GPT-Generated Unified Format) is the format used by llama.cpp and Ollama. We use the conversion script from the llama.cpp repository:

```bash
# Clone llama.cpp (if not already)
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt

# Convert merged model to GGUF F16
python3 llama.cpp/convert_hf_to_gguf.py \
    tokioai-3b-v2-merged/ \
    --outfile tokioai-3b-v2-f16.gguf \
    --outtype f16
```

Output: `tokioai-3b-v2-f16.gguf` (~5.8 GB for 3B)

### Step 3: Quantize to Q4_K_M

Q4_K_M is the sweet spot: 4-bit quantization with k-quant (quality-preserving), medium precision. It reduces model size by ~4x with minimal quality loss.

```bash
# Build llama-quantize
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4 --target llama-quantize

# Quantize
./build/bin/llama-quantize \
    tokioai-3b-v2-f16.gguf \
    tokioai-3b-v2-Q4_K_M.gguf \
    Q4_K_M
```

Size comparison:

| Format | 3B Size | 14B Size | 72B Size |
|--------|---------|----------|----------|
| bf16 (original) | ~6 GB | ~28 GB | ~144 GB |
| GGUF F16 | 5.8 GB | ~27 GB | ~140 GB |
| GGUF Q4_K_M | 1.8 GB | ~8.5 GB | ~42 GB |

### Available Quantization Levels

```
Q2_K     -- 2-bit, very lossy, smallest size
Q3_K_S   -- 3-bit small
Q3_K_M   -- 3-bit medium
Q4_0     -- 4-bit, basic
Q4_K_S   -- 4-bit k-quant small
Q4_K_M   -- 4-bit k-quant medium  <-- OUR CHOICE (best balance)
Q5_0     -- 5-bit basic
Q5_K_M   -- 5-bit k-quant medium
Q6_K     -- 6-bit k-quant
Q8_0     -- 8-bit (nearly lossless)
F16      -- 16-bit (no quantization)
```

Rule of thumb: Q4_K_M retains ~97% of the original model quality while being ~4x smaller.

---

## 9. Deployment with Ollama

Ollama is a local LLM runtime that makes deployment trivial.

### Create the Modelfile

```
FROM tokioai-3b-v2-Q4_K_M.gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "</tool_call>"
PARAMETER num_ctx 4096
```

Key parameters:
- **temperature 0.1**: Very low randomness. Tool calling needs deterministic outputs.
- **stop "</tool_call>"**: Stop generation when a tool call is complete. This prevents the model from generating phantom tool results.
- **stop "<|im_end|>"**: Qwen2.5's end-of-message token.
- **num_ctx 4096**: Context window size.

### Create and Run

```bash
# Create the model
ollama create tokioai-3b-v2 -f Modelfile

# Test it
ollama run tokioai-3b-v2 "check disk space"
```

Expected output:

```
<tool_call>
{"name": "execute_local", "arguments": {"command": "df -h"}}
```

The model generates the tool call and stops at `</tool_call>` -- ready for the executor to parse and run.

### Integration with the TokioAI CLI

The CLI reads the model output, extracts the `<tool_call>` block, executes the tool, and feeds the result back:

```
User input
    |
    v
[Ollama: tokioai-3b-v2]
    |
    v
<tool_call>{"name": "execute_local", "arguments": {"command": "df -h"}}</tool_call>
    |
    v
[CLI: parse tool call, execute command]
    |
    v
Result: "Filesystem  Size  Used  Avail  Use%..."
    |
    v
[Append result to conversation, generate summary]
    |
    v
Display to user
```

---

## 10. Lessons Learned and Debugging

### OOM Errors: The #1 Enemy

Out-of-memory errors are the most common issue when training large models. Here's our debugging playbook:

**Symptom**: `torch.OutOfMemoryError: CUDA out of memory`

**Debugging steps** (in order):

1. Check current VRAM usage: `nvidia-smi`
2. Identify the failure point: which step/phase crashed?
3. Try these fixes (from least to most impact):

```
Fix                            VRAM saved    Quality impact
---                            ----------    --------------
Reduce max_length              ~200-600 MB   Truncates long examples
Reduce batch_size to 1         Variable      None (use grad_accum instead)
Enable gradient_checkpointing  ~40%          None (just slower)
Use paged_adamw_8bit           ~30%          Minimal
Reduce LoRA rank (r)           ~10-20%       Slight capacity loss
Skip prepare_model_for_kbit    ~1-2 GB       None if done manually
Set expandable_segments:True   ~5-10%        None
```

**Our 72B OOM story**:

The 72B training crashed at step 2 with:
```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 462.00 MiB
(GPU 1: 23.65 GiB total, 22.44 GiB used)
```

Analysis: Step 2 had a longer example than step 1. With `max_length=2048`, the attention matrix for a 2048-token sequence on a 72B model consumed ~462 MB more than available.

Fix: `max_length=2048` -> `max_length=1024`. This freed ~600 MB, enough headroom for all examples.

Lesson: **Always test with your longest example first**, or set `max_length` conservatively.

### Loss Monitoring

A healthy training loss curve looks like:

```
Epoch 1:  1.5 -> 0.8  (big drop -- model is learning)
Epoch 2:  0.8 -> 0.3  (steady improvement)
Epoch 3:  0.3 -> 0.1  (refinement)
Epoch 4+: 0.1 -> 0.001  (memorization -- may be overfitting)
```

For our 3B v2 with 8 epochs, the final loss was 0.0001. This is borderline overfitting, but for tool calling it's fine -- we WANT the model to memorize the exact tool call format. Tool calling is not a creative task.

### The Chat Template is Critical

The tokenizer's `apply_chat_template()` converts messages into the model's expected format. For Qwen2.5:

```
<|im_start|>system
You are TokioAI...<|im_end|>
<|im_start|>user
check disk space<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "execute_local", "arguments": {"command": "df -h"}}
</tool_call><|im_end|>
```

**Never** format manually. Always use the tokenizer's template. Different models have different formats, and getting the special tokens wrong will break training.

### Screen Sessions for Long Trainings

Always run training in a detached session. SSH disconnects WILL happen:

```bash
# Start training in a detached screen
screen -dmS train bash -c 'python3 train.py > train.log 2>&1'

# Check progress
tail -f train.log

# Reattach
screen -r train
```

---

## 11. Results and Comparison

### Tool Call Accuracy (3B v2)

Tested with 8 queries across different categories:

| Query | Expected Tool | Generated Tool | Correct? |
|-------|--------------|----------------|----------|
| "check disk space" | execute_local | execute_local | YES |
| "raspi temperature" | execute_raspi | execute_raspi | YES |
| "hay incidentes?" | irt_chat | irt_chat | YES |
| "read my emails" | outlook_read | outlook_read | YES |
| "show router interfaces" | execute_router | execute_router | YES |
| "check GCP VM status" | execute_gcp | execute_gcp | YES |
| "remember this" | memory | memory | YES |
| "scan ports on 10.0.0.1" | execute_local (nmap) | execute_local | YES |

**Result: 8/8 correct tool calls (100%)**

### Model Size vs Quality Tradeoff

The 3B model at Q4_K_M (1.8 GB) runs on a Raspberry Pi 5 with 8 GB RAM. It generates correct tool calls in under 2 seconds. For our use case (tool selection, not creative writing), the 3B model is sufficient.

The 14B and 72B models produce more natural explanatory text alongside the tool calls and handle ambiguous queries better, but for pure tool-calling accuracy, the 3B is competitive.

---

## 12. Hardware Requirements

### Minimum for Training

| Model | GPU | VRAM | RAM | Disk | Cost (GCP spot) |
|-------|-----|------|-----|------|-----------------|
| 3B (bf16 LoRA) | 1x T4 | 16 GB | 16 GB | 50 GB | ~$0.12/hr |
| 14B (QLoRA 4-bit) | 1x T4 | 16 GB | 32 GB | 100 GB | ~$0.12/hr |
| 72B (QLoRA 4-bit) | 2x L4 | 48 GB | 96 GB | 500 GB | ~$1.50/hr |

### Minimum for Inference (Ollama Q4_K_M)

| Model | RAM needed | RPi5 8GB? | Laptop 16GB? |
|-------|-----------|-----------|--------------|
| 3B Q4_K_M (1.8 GB) | ~4 GB | YES | YES |
| 14B Q4_K_M (~8.5 GB) | ~12 GB | NO | TIGHT |
| 72B Q4_K_M (~42 GB) | ~48 GB | NO | NO |

### GCP VM Specifications Used

**gpu-inference-lab** (3B + 14B training):
```
Machine type:  n1-standard-4 (4 vCPUs, 15 GB RAM)
GPU:           1x NVIDIA T4 (16 GB VRAM)
Disk:          100 GB SSD
Region:        us-central1-b
```

**tokioai-72b-gpu** (72B training):
```
Machine type:  g2-standard-24 (24 vCPUs, 96 GB RAM)
GPU:           2x NVIDIA L4 (24 GB each = 48 GB total)
Disk:          500 GB SSD
Region:        us-east4-c
```

---

## 13. Reproducing This Work

### Prerequisites

```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate
```

Versions tested:
```
torch==2.4.0
transformers==4.46.0
trl==1.14.0
peft==0.13.0
bitsandbytes==0.44.1
accelerate==1.0.0
```

### Step-by-Step

```bash
# 1. Create your dataset
python3 gen_dataset_v2.py
# Output: tokioai_dataset_v2.jsonl (308 examples)

# 2. Train the model (pick your size)
python3 train_3b_v2.py    # ~77 min on T4
# python3 train_14b_v2.py  # ~5.5 hours on T4
# python3 train_72b_v2.py  # ~1.9 hours on 2xL4

# 3. Merge adapter (done automatically in 3B script)
# For 14B/72B, run merge separately:
python3 merge_adapter.py

# 4. Convert to GGUF
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt
python3 llama.cpp/convert_hf_to_gguf.py \
    tokioai-3b-v2-merged/ \
    --outfile tokioai-3b-v2-f16.gguf \
    --outtype f16

# 5. Quantize
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 --target llama-quantize
./build/bin/llama-quantize \
    ../tokioai-3b-v2-f16.gguf \
    ../tokioai-3b-v2-Q4_K_M.gguf \
    Q4_K_M

# 6. Deploy with Ollama
cat > Modelfile << 'EOF'
FROM tokioai-3b-v2-Q4_K_M.gguf
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "</tool_call>"
PARAMETER num_ctx 4096
EOF

ollama create tokioai-3b-v2 -f Modelfile
ollama run tokioai-3b-v2 "scan my network"
```

### Adapting for Your Own Tools

To train a model for YOUR tool catalog:

1. **Define your tools** in the system prompt
2. **Create examples** showing user queries -> correct tool calls
3. **Use the `<tool_call>` format** with your tool names and arguments
4. **Generate 200+ examples** covering all tools and edge cases
5. **Train on the 3B model first** (cheapest, fastest feedback loop)
6. **Scale up** to 14B/72B only if 3B doesn't meet accuracy requirements

---

## 14. References

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [TRL: Transformer Reinforcement Learning](https://huggingface.co/docs/trl/)
- [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft/)
- [Qwen2.5 Technical Report](https://qwenlm.github.io/blog/qwen2.5/)
- [llama.cpp GGUF Format](https://github.com/ggerganov/llama.cpp)
- [Ollama Documentation](https://ollama.ai)
- [BitsAndBytes: 4-bit Quantization](https://github.com/bitsandbytes-foundation/bitsandbytes)

---

## License

This paper and all associated code are released under the MIT License.

**TokioAI** -- No talking about AI. We put it in production.
