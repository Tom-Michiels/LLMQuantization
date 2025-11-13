# LLM Quantization Experiment

Experiments with quantized Llama models and RMSNorm weight precision.

## Overview

This project downloads quantized versions of Llama 3.2 1B in different quantization formats (Q4_0, Q4_1, Q4_K_S), measures their perplexity on WikiText-2, then modifies them to round RMSNorm fp32 weights to fp16 and re-measures perplexity to understand the impact.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download models (see USAGE.md if this fails due to network restrictions)
python3 download_models.py

# Run the full experiment
bash run_experiment.sh
```

**Note:** If model download fails due to network restrictions, see [USAGE.md](USAGE.md) for manual download instructions.

This will:
1. Download quantized Llama 3.2 1B models (Q4_0, Q4_1, Q4_K_S)
2. Setup llama.cpp for perplexity measurement
3. Download WikiText-2 dataset
4. Measure baseline perplexity
5. Modify models (round RMSNorm weights to fp16)
6. Re-measure perplexity

## Manual Steps

### 1. Download Models

```bash
python3 download_models.py
```

Downloads from HuggingFace: `bartowski/Llama-3.2-1B-GGUF`
- Llama-3.2-1B-Q4_0.gguf
- Llama-3.2-1B-Q4_1.gguf
- Llama-3.2-1B-Q4_K_S.gguf

### 2. Setup llama.cpp

```bash
bash setup_llamacpp.sh
```

Clones and builds llama.cpp, downloads WikiText-2.

### 3. Measure Baseline Perplexity

```bash
./llama.cpp/perplexity -m models/Llama-3.2-1B-Q4_0.gguf -f wikitext-2/wiki.test.raw
```

Or use the Python wrapper:

```bash
python3 measure_perplexity.py
```

### 4. Modify RMSNorm Weights

```bash
python3 modify_rmsnorm.py
```

This reads each GGUF file, finds all RMSNorm weight tensors (fp32), rounds them to fp16 and back to fp32, and saves modified models to `models/modified/`.

### 5. Compare Results

The measurement script automatically compares original vs modified models and displays:
- Perplexity for each model
- Difference between original and fp16-RMSNorm versions
- Average impact across all quantization types

## Project Structure

```
.
├── download_models.py       # Download quantized models from HF
├── modify_rmsnorm.py         # Modify GGUF to round RMSNorm weights
├── measure_perplexity.py     # Measure and compare perplexity
├── setup_llamacpp.sh         # Setup llama.cpp and WikiText-2
├── run_experiment.sh         # Master script for full experiment
├── requirements.txt          # Python dependencies
├── models/                   # Downloaded models
│   └── modified/            # Modified models (fp16 RMSNorm)
├── results/                  # Experiment results
└── llama.cpp/               # llama.cpp repository (cloned)
```

## Results Format

Results are saved to `results/perplexity_results.json`:

```json
{
  "original": {
    "Llama-3.2-1B-Q4_0.gguf": 12.34,
    "Llama-3.2-1B-Q4_1.gguf": 12.45,
    "Llama-3.2-1B-Q4_K_S.gguf": 12.23
  },
  "modified": {
    "modified_Llama-3.2-1B-Q4_0.gguf": 12.36,
    "modified_Llama-3.2-1B-Q4_1.gguf": 12.47,
    "modified_Llama-3.2-1B-Q4_K_S.gguf": 12.25
  }
}
```

## Requirements

- Python 3.7+
- pip packages: huggingface_hub, datasets, numpy, gguf
- Build tools: make, gcc/clang (for llama.cpp)
- ~5GB disk space for models
- ~8GB RAM for inference

## How It Works

### RMSNorm Weight Modification

The RMSNorm layer in transformers uses fp32 weights to scale features. This experiment tests the impact of reducing these weights to fp16 precision:

1. Load GGUF model file
2. Identify RMSNorm weight tensors (names contain "norm.weight")
3. Convert fp32 → fp16 → fp32 (simulating precision loss)
4. Save modified GGUF file
5. Compare perplexity

### Perplexity Measurement

Uses llama.cpp's built-in perplexity tool which:
- Loads the GGUF model
- Tokenizes WikiText-2 test set
- Calculates log probability of each token
- Reports perplexity = exp(negative log likelihood)

## Expected Results

RMSNorm weights affect the scale of activations flowing through the network. Reducing precision from fp32 to fp16 may:
- Slightly increase perplexity (worse language modeling)
- Have minimal impact if the quantization already dominates errors
- Vary by quantization type (Q4_0 vs Q4_1 vs Q4_K_S)

## License

MIT
