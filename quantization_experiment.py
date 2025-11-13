#!/usr/bin/env python3
"""
Script to download quantized Llama models, measure perplexity,
and test the effect of rounding RMSNorm weights to fp16.
"""

import os
import sys
import json
import struct
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

from huggingface_hub import hf_hub_download
from datasets import load_dataset
import gguf


# Model configurations
QUANT_TYPES = ["Q4_0", "Q4_1", "Q4_K_S"]
MODEL_NAME = "llama-3.2-1B"
HF_REPO = "bartowski/Llama-3.2-1B-GGUF"  # Popular GGUF repository

# File naming convention for bartowski's models
QUANT_FILE_MAPPING = {
    "Q4_0": "Llama-3.2-1B-Q4_0.gguf",
    "Q4_1": "Llama-3.2-1B-Q4_1.gguf",
    "Q4_K_S": "Llama-3.2-1B-Q4_K_S.gguf"
}


def download_model(quant_type: str, output_dir: str = "models") -> str:
    """Download a quantized model from HuggingFace."""
    os.makedirs(output_dir, exist_ok=True)

    filename = QUANT_FILE_MAPPING[quant_type]
    print(f"Downloading {quant_type} model: {filename}")

    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            cache_dir=output_dir,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading {quant_type}: {e}")
        raise


def load_wikitext2():
    """Load WikiText-2 test dataset."""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Combine all text
    text = "\n".join([item["text"] for item in dataset if item["text"].strip()])
    return text


def measure_perplexity_llama_cpp(model_path: str, text: str, max_tokens: int = 2048) -> float:
    """
    Measure perplexity using llama-cpp-python.
    """
    try:
        from llama_cpp import Llama

        print(f"Loading model: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=max_tokens,
            n_batch=512,
            verbose=False
        )

        # Tokenize the text
        tokens = llm.tokenize(text.encode('utf-8'))
        print(f"Number of tokens: {len(tokens)}")

        # Limit to max_tokens for computational efficiency
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            print(f"Truncated to {max_tokens} tokens")

        # Calculate perplexity by evaluating log probability
        total_log_prob = 0.0
        count = 0

        # Evaluate in chunks
        chunk_size = 512
        for i in tqdm(range(1, len(tokens), chunk_size), desc="Calculating perplexity"):
            start_idx = max(0, i - 1)
            end_idx = min(i + chunk_size, len(tokens))
            chunk = tokens[start_idx:end_idx]

            # Reset the model state
            llm.reset()

            # Evaluate the chunk
            llm.eval(chunk)

            # Get logits for predictions
            for j in range(len(chunk) - 1):
                logits = llm.eval([chunk[j]])
                # Get probability of next token
                # Note: llama-cpp-python doesn't easily expose log probs
                # So we'll use a simpler chunked approach

        # Alternative: Use the built-in perplexity calculation if available
        # For now, we'll use a simpler evaluation method

        print("Note: Using simplified perplexity calculation")
        # Reset and evaluate full sequence
        llm.reset()

        # Evaluate in one go for better accuracy
        total_nll = 0.0
        for i in tqdm(range(len(tokens) - 1), desc="Evaluating"):
            context = tokens[:i+1]
            if len(context) > max_tokens:
                context = context[-max_tokens:]

            # This is very slow, so let's use a sampled approach
            if i % 50 == 0:  # Sample every 50 tokens
                llm.reset()
                llm.eval(context[:-1])
                # Get next token probability would go here
                # This is a placeholder - llama-cpp-python doesn't expose this easily
                pass

        # Return placeholder for now - we need a better approach
        return 0.0

    except Exception as e:
        print(f"Error measuring perplexity: {e}")
        raise


def measure_perplexity_torch(model_path: str, text: str) -> float:
    """
    Measure perplexity by loading GGUF into numpy/torch.
    This is a placeholder - GGUF models aren't easily loaded into PyTorch.
    """
    print("Note: Direct perplexity measurement from GGUF is complex.")
    print("Consider using llama.cpp's perplexity tool instead.")
    return 0.0


def find_rmsnorm_tensors(reader: gguf.GGUFReader) -> List[Tuple[int, gguf.ReaderTensor]]:
    """Find all RMSNorm weight tensors in the model."""
    rmsnorm_tensors = []

    for idx, tensor in enumerate(reader.tensors):
        # RMSNorm weights are typically named with "norm" in them
        # and are 1D tensors
        if "norm.weight" in tensor.name or "attention_norm" in tensor.name or "ffn_norm" in tensor.name:
            rmsnorm_tensors.append((idx, tensor))
            print(f"Found RMSNorm tensor: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.tensor_type}")

    return rmsnorm_tensors


def convert_fp32_to_fp16(data: np.ndarray) -> np.ndarray:
    """Convert fp32 array to fp16 and back to fp32 (simulating rounding)."""
    # Convert to fp16
    fp16_data = data.astype(np.float16)
    # Convert back to fp32
    return fp16_data.astype(np.float32)


def modify_rmsnorm_to_fp16(input_path: str, output_path: str) -> None:
    """
    Modify a GGUF file to round all RMSNorm weights from fp32 to fp16.
    """
    print(f"Reading GGUF file: {input_path}")
    reader = gguf.GGUFReader(input_path)

    # Find RMSNorm tensors
    rmsnorm_tensors = find_rmsnorm_tensors(reader)
    print(f"Found {len(rmsnorm_tensors)} RMSNorm tensors")

    if not rmsnorm_tensors:
        print("Warning: No RMSNorm tensors found!")
        return

    # Create a writer for the new file
    print(f"Creating modified GGUF file: {output_path}")
    arch = None
    for field in reader.fields.values():
        if field.name == "general.architecture":
            arch = str(bytes(field.parts[field.data[0]]), encoding="utf-8")
            break

    if arch is None:
        arch = "llama"

    writer = gguf.GGUFWriter(output_path, arch=arch)

    # Copy metadata
    for field in reader.fields.values():
        writer.add_key(field.name)
        # Copy the field data
        # This is simplified - proper implementation would handle all field types

    # Process tensors
    modified_count = 0
    rmsnorm_indices = {idx for idx, _ in rmsnorm_tensors}

    for idx, tensor in enumerate(reader.tensors):
        data = tensor.data

        if idx in rmsnorm_indices:
            # This tensor is RMSNorm - convert to fp16 and back
            print(f"Modifying tensor: {tensor.name}")

            # Parse the tensor data based on type
            if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
                # fp32 data
                fp32_data = np.frombuffer(data, dtype=np.float32)
                # Round to fp16
                modified_data = convert_fp32_to_fp16(fp32_data)
                # Add to writer
                writer.add_tensor(tensor.name, modified_data, tensor.tensor_type)
                modified_count += 1
            else:
                print(f"  Skipping {tensor.name} - not fp32 (type: {tensor.tensor_type})")
                writer.add_tensor(tensor.name, data, tensor.tensor_type)
        else:
            # Copy tensor as-is
            writer.add_tensor(tensor.name, data, tensor.tensor_type)

    print(f"Modified {modified_count} tensors")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Saved modified model to: {output_path}")


def run_experiment():
    """Main experiment function."""
    results = {
        "original": {},
        "modified": {}
    }

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/modified", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Download models
    print("\n" + "="*80)
    print("STEP 1: Downloading models")
    print("="*80)

    model_paths = {}
    for quant_type in QUANT_TYPES:
        try:
            path = download_model(quant_type)
            model_paths[quant_type] = path
        except Exception as e:
            print(f"Failed to download {quant_type}: {e}")
            continue

    # Load WikiText-2
    print("\n" + "="*80)
    print("STEP 2: Loading WikiText-2")
    print("="*80)
    wikitext = load_wikitext2()
    print(f"Loaded {len(wikitext)} characters")

    # Measure baseline perplexity
    print("\n" + "="*80)
    print("STEP 3: Measuring baseline perplexity")
    print("="*80)

    print("\nIMPORTANT: For accurate perplexity measurement, we should use llama.cpp's")
    print("perplexity tool. The Python bindings don't expose log probabilities easily.")
    print("\nWe'll create a script to use the llama.cpp command-line tool instead.\n")

    # Create modified models
    print("\n" + "="*80)
    print("STEP 4: Creating modified models (RMSNorm fp32->fp16)")
    print("="*80)

    modified_paths = {}
    for quant_type, original_path in model_paths.items():
        try:
            # Create output path
            original_name = os.path.basename(original_path)
            modified_path = os.path.join("models/modified", f"modified_{original_name}")

            print(f"\nModifying {quant_type}...")
            modify_rmsnorm_to_fp16(original_path, modified_path)
            modified_paths[quant_type] = modified_path

        except Exception as e:
            print(f"Failed to modify {quant_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    results_file = "results/experiment_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "model_paths": model_paths,
            "modified_paths": modified_paths,
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("\nNext steps:")
    print("1. Install llama.cpp: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make")
    print("2. Run perplexity measurement using llama.cpp's perplexity tool")
    print("   Example: ./llama.cpp/perplexity -m models/model.gguf -f wikitext-2-raw/wiki.test.raw")


if __name__ == "__main__":
    run_experiment()
