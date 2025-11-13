#!/bin/bash
# Setup llama.cpp for perplexity measurement

set -e

echo "================================"
echo "Setting up llama.cpp"
echo "================================"

# Clone llama.cpp if not already present
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
else
    echo "llama.cpp directory already exists"
    cd llama.cpp
    echo "Pulling latest changes..."
    git pull
fi

# Build llama.cpp
echo ""
echo "Building llama.cpp..."
make clean
make -j$(nproc)

# Verify perplexity tool was built
if [ -f "perplexity" ]; then
    echo ""
    echo "✓ llama.cpp built successfully!"
    echo "✓ Perplexity tool is ready"
else
    echo ""
    echo "✗ Error: perplexity tool not found"
    exit 1
fi

cd ..

# Download WikiText-2 dataset if not present
if [ ! -d "wikitext-2" ]; then
    echo ""
    echo "Downloading WikiText-2 dataset..."
    mkdir -p wikitext-2
    cd wikitext-2

    # Download from HuggingFace or use Python to prepare it
    echo "Preparing WikiText-2 test data..."
    cd ..

    python3 << 'EOF'
from datasets import load_dataset
import os

os.makedirs("wikitext-2", exist_ok=True)

print("Loading WikiText-2 test split...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# Save test data
with open("wikitext-2/wiki.test.raw", "w") as f:
    for item in dataset:
        f.write(item["text"] + "\n")

print("✓ WikiText-2 test data saved to wikitext-2/wiki.test.raw")
EOF

else
    echo "WikiText-2 dataset already present"
fi

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "You can now measure perplexity using:"
echo "  ./llama.cpp/perplexity -m <model.gguf> -f wikitext-2/wiki.test.raw"
