#!/usr/bin/env python3
"""
Download popular Llama models with Q4_0, Q4_1, or Q4_K_S quantization from Hugging Face
"""

import os
import sys
from huggingface_hub import hf_hub_download

MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Model configurations: (repo_id, filename, description)
MODELS = {
    "llama-3.2-1b-q4ks": (
        "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_S.gguf",
        "Llama 3.2 1B Instruct (Q4_K_S, 756 MB)"
    ),
    "llama-3.2-3b-q4ks": (
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-Q4_K_S.gguf",
        "Llama 3.2 3B Instruct (Q4_K_S, 1.95 GB)"
    ),
    "llama-2-7b-q4ks": (
        "TheBloke/Llama-2-7B-GGUF",
        "llama-2-7b.Q4_K_S.gguf",
        "Llama 2 7B (Q4_K_S, 3.56 GB)"
    ),
    "llama-2-7b-chat-q4ks": (
        "TheBloke/Llama-2-7B-Chat-GGUF",
        "llama-2-7b-chat.Q4_K_S.gguf",
        "Llama 2 7B Chat (Q4_K_S, 3.56 GB)"
    ),
    "llama-2-7b-q40": (
        "TheBloke/Llama-2-7B-GGUF",
        "llama-2-7b.Q4_0.gguf",
        "Llama 2 7B (Q4_0, 3.50 GB)"
    ),
    "llama-3-8b-q4ks": (
        "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
        "Meta-Llama-3-8B-Instruct-Q4_K_S.gguf",
        "Llama 3 8B Instruct (Q4_K_S, 4.69 GB)"
    ),
    "llama-3-8b-q40": (
        "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
        "Meta-Llama-3-8B-Instruct-Q4_0.gguf",
        "Llama 3 8B Instruct (Q4_0, 4.33 GB)"
    ),
    "llama-3.1-8b-q4ks": (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf",
        "Llama 3.1 8B Instruct (Q4_K_S, 4.92 GB)"
    ),
    "llama-2-13b-q4ks": (
        "TheBloke/Llama-2-13B-GGUF",
        "llama-2-13b.Q4_K_S.gguf",
        "Llama 2 13B (Q4_K_S, 7.16 GB)"
    ),
}

def download_model(model_key):
    """Download a single model"""
    if model_key not in MODELS:
        print(f"❌ Unknown model: {model_key}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False

    repo_id, filename, description = MODELS[model_key]

    print(f"\n{'='*60}")
    print(f"📥 Downloading: {description}")
    print(f"Repository: {repo_id}")
    print(f"File: {filename}")
    print(f"{'='*60}\n")

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"\n✅ Successfully downloaded to: {local_path}")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {filename}: {e}")
        return False

def print_usage():
    """Print usage information"""
    print("Usage: python3 download_models.py [model_key]")
    print("\nAvailable models:")
    for key, (_, _, desc) in MODELS.items():
        print(f"  • {key:20s} - {desc}")
    print("\nSpecial options:")
    print("  • all-small          - Download all small models (<2GB)")
    print("  • all-7b             - Download all 7B/8B models")
    print("  • all                - Download all models")
    print("\nExample: python3 download_models.py llama-3.2-1b-q4ks")

def main():
    if len(sys.argv) < 2:
        print_usage()
        return

    option = sys.argv[1]

    if option == "all-small":
        models = ["llama-3.2-1b-q4ks", "llama-3.2-3b-q4ks"]
    elif option == "all-7b":
        models = ["llama-2-7b-q4ks", "llama-2-7b-chat-q4ks", "llama-3-8b-q4ks", "llama-3.1-8b-q4ks"]
    elif option == "all":
        models = list(MODELS.keys())
    else:
        models = [option]

    print(f"\n🚀 Starting download of {len(models)} model(s)...")

    success_count = 0
    for model_key in models:
        if download_model(model_key):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"✨ Download complete: {success_count}/{len(models)} successful")
    print(f"Models saved to: {os.path.abspath(MODELS_DIR)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
