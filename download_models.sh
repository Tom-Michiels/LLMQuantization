#!/bin/bash

# Script to download popular Llama models with Q4_0, Q4_1, or Q4_K_S quantization
# Usage: ./download_models.sh [model_name]
# If no model name is provided, it will show available options

set -e

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [model_name]"
    echo ""
    echo "Available models:"
    echo "  1) llama-2-7b-q4ks         - Llama 2 7B (Q4_K_S, 3.56 GB)"
    echo "  2) llama-2-7b-chat-q4ks    - Llama 2 7B Chat (Q4_K_S, 3.56 GB)"
    echo "  3) llama-2-7b-q40          - Llama 2 7B (Q4_0, 3.50 GB)"
    echo "  4) llama-2-13b-q4ks        - Llama 2 13B (Q4_K_S, 7.16 GB)"
    echo "  5) llama-3-8b-q4ks         - Llama 3 8B Instruct (Q4_K_S, 4.69 GB)"
    echo "  6) llama-3-8b-q40          - Llama 3 8B Instruct (Q4_0, 4.33 GB)"
    echo "  7) llama-3.1-8b-q4ks       - Llama 3.1 8B Instruct (Q4_K_S, 4.92 GB)"
    echo "  8) llama-3.2-3b-q4ks       - Llama 3.2 3B Instruct (Q4_K_S, 1.95 GB)"
    echo "  9) llama-3.2-1b-q4ks       - Llama 3.2 1B Instruct (Q4_K_S, 756 MB)"
    echo "  10) all-small              - Download all small models (<2GB)"
    echo "  11) all-7b                 - Download all 7B/8B models"
    echo ""
    echo "Example: $0 llama-3-8b-q4ks"
}

download_model() {
    local url=$1
    local filename=$2
    local description=$3

    echo -e "${GREEN}Downloading: $description${NC}"
    echo "URL: $url"
    echo "Destination: $MODELS_DIR/$filename"
    echo ""

    if [ -f "$MODELS_DIR/$filename" ]; then
        echo -e "${YELLOW}File already exists. Using -c flag to resume if incomplete.${NC}"
    fi

    wget -c "$url" -O "$MODELS_DIR/$filename" --progress=bar:force

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully downloaded: $filename${NC}"
        echo ""
    else
        echo -e "${RED}✗ Failed to download: $filename${NC}"
        echo ""
        return 1
    fi
}

# Model URLs
case "$1" in
    "llama-2-7b-q4ks")
        download_model \
            "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_S.gguf" \
            "llama-2-7b.Q4_K_S.gguf" \
            "Llama 2 7B (Q4_K_S)"
        ;;

    "llama-2-7b-chat-q4ks")
        download_model \
            "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_S.gguf" \
            "llama-2-7b-chat.Q4_K_S.gguf" \
            "Llama 2 7B Chat (Q4_K_S)"
        ;;

    "llama-2-7b-q40")
        download_model \
            "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf" \
            "llama-2-7b.Q4_0.gguf" \
            "Llama 2 7B (Q4_0)"
        ;;

    "llama-2-13b-q4ks")
        download_model \
            "https://huggingface.co/TheBloke/Llama-2-13B-GGUF/resolve/main/llama-2-13b.Q4_K_S.gguf" \
            "llama-2-13b.Q4_K_S.gguf" \
            "Llama 2 13B (Q4_K_S)"
        ;;

    "llama-3-8b-q4ks")
        download_model \
            "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_S.gguf" \
            "Meta-Llama-3-8B-Instruct-Q4_K_S.gguf" \
            "Llama 3 8B Instruct (Q4_K_S)"
        ;;

    "llama-3-8b-q40")
        download_model \
            "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_0.gguf" \
            "Meta-Llama-3-8B-Instruct-Q4_0.gguf" \
            "Llama 3 8B Instruct (Q4_0)"
        ;;

    "llama-3.1-8b-q4ks")
        download_model \
            "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf" \
            "Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf" \
            "Llama 3.1 8B Instruct (Q4_K_S)"
        ;;

    "llama-3.2-3b-q4ks")
        download_model \
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_S.gguf" \
            "Llama-3.2-3B-Instruct-Q4_K_S.gguf" \
            "Llama 3.2 3B Instruct (Q4_K_S)"
        ;;

    "llama-3.2-1b-q4ks")
        download_model \
            "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_S.gguf" \
            "Llama-3.2-1B-Instruct-Q4_K_S.gguf" \
            "Llama 3.2 1B Instruct (Q4_K_S)"
        ;;

    "all-small")
        echo "Downloading all small models (<2GB)..."
        $0 llama-3.2-1b-q4ks
        $0 llama-3.2-3b-q4ks
        ;;

    "all-7b")
        echo "Downloading all 7B/8B models..."
        $0 llama-2-7b-q4ks
        $0 llama-2-7b-chat-q4ks
        $0 llama-3-8b-q4ks
        $0 llama-3.1-8b-q4ks
        ;;

    "")
        print_usage
        exit 0
        ;;

    *)
        echo -e "${RED}Unknown model: $1${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac

echo -e "${GREEN}Done!${NC}"
echo "Models are stored in: $MODELS_DIR"
