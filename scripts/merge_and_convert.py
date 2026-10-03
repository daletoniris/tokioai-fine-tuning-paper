#!/usr/bin/env python3
"""
Post-training pipeline: Merge LoRA adapter + Convert to GGUF + Quantize + Ollama

Usage:
    python3 merge_and_convert.py --model 3b --adapter tokioai-3b-v2/
    python3 merge_and_convert.py --model 14b --adapter tokioai-14b-v2/
    python3 merge_and_convert.py --model 72b --adapter tokioai-72b-v2/ --skip-merge
        (72B merge needs ~150 GB RAM; if not available, skip and use adapter directly)
"""
import argparse
import os
import subprocess
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Model name mapping
MODELS = {
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "14b": "Qwen/Qwen2.5-14B-Instruct",
    "72b": "Qwen/Qwen2.5-72B-Instruct",
}


def merge_adapter(model_name: str, adapter_dir: str, output_dir: str):
    """Merge LoRA adapter weights into the base model."""
    print(f"[{datetime.now()}] Loading base model on CPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"[{datetime.now()}] Loading adapter from {adapter_dir}...")
    merged = PeftModel.from_pretrained(base_model, adapter_dir)

    print(f"[{datetime.now()}] Merging (W_merged = W_base + A * B)...")
    merged = merged.merge_and_unload()

    print(f"[{datetime.now()}] Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    del merged, base_model
    print(f"Merged model saved to {output_dir}/")


def convert_to_gguf(merged_dir: str, gguf_dir: str, model_tag: str):
    """Convert HuggingFace model to GGUF F16 format."""
    llama_cpp_dir = os.path.expanduser("~/llama.cpp")
    if not os.path.exists(llama_cpp_dir):
        print("Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp", llama_cpp_dir],
            timeout=120,
        )
        subprocess.run(
            ["pip", "install", "-r", f"{llama_cpp_dir}/requirements.txt"],
            capture_output=True, timeout=120,
        )

    os.makedirs(gguf_dir, exist_ok=True)
    f16_path = os.path.join(gguf_dir, f"tokioai-{model_tag}-v2-f16.gguf")

    print(f"[{datetime.now()}] Converting to GGUF F16...")
    subprocess.run(
        [
            "python3", f"{llama_cpp_dir}/convert_hf_to_gguf.py",
            merged_dir, "--outfile", f16_path, "--outtype", "f16",
        ],
        timeout=1800,  # 30 min for large models
    )

    if os.path.exists(f16_path):
        print(f"GGUF F16: {os.path.getsize(f16_path) / 1e9:.1f} GB -> {f16_path}")
    return f16_path


def quantize(f16_path: str, gguf_dir: str, model_tag: str, quant_type: str = "Q4_K_M"):
    """Quantize GGUF F16 to a smaller format."""
    llama_cpp_dir = os.path.expanduser("~/llama.cpp")

    # Build quantize tool
    print(f"[{datetime.now()}] Building llama-quantize...")
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
    if not os.path.exists(quantize_bin):
        print("ERROR: llama-quantize not found. Check llama.cpp build.")
        return None

    q_path = os.path.join(gguf_dir, f"tokioai-{model_tag}-v2-{quant_type}.gguf")

    print(f"[{datetime.now()}] Quantizing to {quant_type}...")
    subprocess.run([quantize_bin, f16_path, q_path, quant_type], timeout=1800)

    if os.path.exists(q_path):
        print(f"{quant_type}: {os.path.getsize(q_path) / 1e9:.1f} GB -> {q_path}")
    return q_path


def create_ollama_model(gguf_path: str, model_tag: str):
    """Register the quantized model in Ollama."""
    model_name = f"tokioai-{model_tag}-v2"
    modelfile_path = f"Modelfile-{model_tag}-v2"

    with open(modelfile_path, "w") as f:
        f.write(f"FROM {os.path.abspath(gguf_path)}\n")
        f.write("PARAMETER temperature 0.1\n")
        f.write("PARAMETER top_p 0.9\n")
        f.write('PARAMETER stop "<|im_end|>"\n')
        f.write('PARAMETER stop "</tool_call>"\n')
        f.write("PARAMETER num_ctx 4096\n")

    print(f"[{datetime.now()}] Creating Ollama model '{model_name}'...")
    subprocess.run(["ollama", "create", model_name, "-f", modelfile_path], timeout=300)
    print(f"Model registered. Test with: ollama run {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Post-training: merge + GGUF + quantize + Ollama")
    parser.add_argument("--model", required=True, choices=["3b", "14b", "72b"])
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merge (use if not enough RAM, e.g., 72B needs ~150 GB)")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--quant-type", default="Q4_K_M",
                        help="Quantization type (default: Q4_K_M)")
    args = parser.parse_args()

    model_name = MODELS[args.model]
    merged_dir = f"tokioai-{args.model}-v2-merged"
    gguf_dir = f"tokioai-{args.model}-v2-gguf"

    # Step 1: Merge
    if not args.skip_merge:
        merge_adapter(model_name, args.adapter, merged_dir)
    else:
        print("Skipping merge (--skip-merge)")
        if not os.path.exists(merged_dir):
            print(f"WARNING: {merged_dir} doesn't exist. GGUF conversion will fail.")

    # Step 2: GGUF
    f16_path = None
    if not args.skip_gguf:
        f16_path = convert_to_gguf(merged_dir, gguf_dir, args.model)
    else:
        print("Skipping GGUF conversion")

    # Step 3: Quantize
    q_path = None
    if not args.skip_quantize and f16_path:
        q_path = quantize(f16_path, gguf_dir, args.model, args.quant_type)
    else:
        # Try to find existing quantized file
        q_path = os.path.join(gguf_dir, f"tokioai-{args.model}-v2-{args.quant_type}.gguf")
        if not os.path.exists(q_path):
            q_path = None

    # Step 4: Ollama
    if not args.skip_ollama and q_path:
        create_ollama_model(q_path, args.model)

    print(f"\n[{datetime.now()}] === Pipeline complete ===")


if __name__ == "__main__":
    main()
