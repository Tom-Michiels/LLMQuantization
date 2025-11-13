#!/bin/bash
# Master script to run the complete experiment

set -e

echo "========================================================================"
echo "LLM Quantization Experiment"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Download Llama 3.2 1B models (Q4_0, Q4_1, Q4_K_S)"
echo "  2. Setup llama.cpp for perplexity measurement"
echo "  3. Measure baseline perplexity on WikiText-2"
echo "  4. Modify models (round RMSNorm weights fp32→fp16)"
echo "  5. Re-measure perplexity on modified models"
echo ""
echo "========================================================================"
echo ""

# Check if Python dependencies are installed
echo "Checking Python dependencies..."
if ! python3 -c "import huggingface_hub, datasets, gguf, numpy" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install -q -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

echo ""
echo "========================================================================"
echo "STEP 1: Downloading Models"
echo "========================================================================"
python3 download_models.py

echo ""
echo "========================================================================"
echo "STEP 2: Setting up llama.cpp"
echo "========================================================================"
bash setup_llamacpp.sh

echo ""
echo "========================================================================"
echo "STEP 3: Measuring Baseline Perplexity"
echo "========================================================================"
echo "(This may take a while...)"

# Measure perplexity for original models
for model in models/*.gguf; do
    if [ -f "$model" ]; then
        echo ""
        echo "Measuring perplexity for: $model"
        ./llama.cpp/perplexity -m "$model" -f wikitext-2/wiki.test.raw -ngl 0 | tee -a results/baseline_perplexity.log
    fi
done

echo ""
echo "========================================================================"
echo "STEP 4: Modifying Models (RMSNorm fp32→fp16)"
echo "========================================================================"
python3 modify_rmsnorm.py

echo ""
echo "========================================================================"
echo "STEP 5: Measuring Modified Model Perplexity"
echo "========================================================================"
echo "(This may take a while...)"

# Measure perplexity for modified models
for model in models/modified/*.gguf; do
    if [ -f "$model" ]; then
        echo ""
        echo "Measuring perplexity for: $model"
        ./llama.cpp/perplexity -m "$model" -f wikitext-2/wiki.test.raw -ngl 0 | tee -a results/modified_perplexity.log
    fi
done

echo ""
echo "========================================================================"
echo "STEP 6: Generating Results"
echo "========================================================================"
python3 measure_perplexity.py

echo ""
echo "========================================================================"
echo "Experiment Complete!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - results/perplexity_results.json"
echo "  - results/baseline_perplexity.log"
echo "  - results/modified_perplexity.log"
echo ""
echo "Check the results above for the comparison between original and"
echo "modified (fp16 RMSNorm) models."
