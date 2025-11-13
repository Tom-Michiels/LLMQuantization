#!/usr/bin/env python3
"""
Download quantized Llama 3.2 1B models from HuggingFace.
"""

import os
import sys
from huggingface_hub import hf_hub_download

# Model configurations
QUANT_TYPES = ["Q4_0", "Q4_1", "Q4_K_S"]
HF_REPO = "QuantFactory/Llama-3.2-1B-GGUF"

# File naming convention (QuantFactory uses this pattern)
QUANT_FILE_MAPPING = {
    "Q4_0": "Llama-3.2-1B.Q4_0.gguf",
    "Q4_1": "Llama-3.2-1B.Q4_1.gguf",
    "Q4_K_S": "Llama-3.2-1B.Q4_K_S.gguf"
}


def download_model(quant_type: str, output_dir: str = "models") -> str:
    """Download a quantized model from HuggingFace."""
    os.makedirs(output_dir, exist_ok=True)

    filename = QUANT_FILE_MAPPING[quant_type]
    print(f"\nDownloading {quant_type} model: {filename}")
    print(f"Repository: {HF_REPO}")

    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"✓ Downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"✗ Error downloading {quant_type}: {e}")
        raise


def main():
    """Download all quantized models."""
    print("="*80)
    print("Downloading Llama 3.2 1B Quantized Models")
    print("="*80)

    model_paths = {}
    for quant_type in QUANT_TYPES:
        try:
            path = download_model(quant_type)
            model_paths[quant_type] = path
        except Exception as e:
            print(f"Failed to download {quant_type}, continuing...")
            continue

    print("\n" + "="*80)
    print("Download Summary")
    print("="*80)
    for quant_type, path in model_paths.items():
        print(f"{quant_type}: {path}")

    print(f"\nSuccessfully downloaded {len(model_paths)}/{len(QUANT_TYPES)} models")

    return model_paths


if __name__ == "__main__":
    main()
