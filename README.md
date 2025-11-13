# LLM Quantization - Llama Models Collection

A collection of popular quantized Llama models in GGUF format, optimized for efficient inference with llama.cpp.

## Overview

This repository provides access to popular Llama models quantized using Q4_0, Q4_1, and Q4_K_S formats. These quantizations reduce model size by ~75% while maintaining good quality, making them suitable for running on consumer hardware.

## Quantization Formats

- **Q4_0**: 4-bit quantization (older method), optimized for ARM chips, ~3.5GB for 7B models
- **Q4_1**: 4-bit quantization (improved quality), ~4GB for 7B models
- **Q4_K_S**: K-quant small format (recommended), 4.5 bits per weight, ~3.56GB for 7B models

## Quick Start

### Download Models

Use the provided script to download models:

```bash
# Show available models
./download_models.sh

# Download a specific model
./download_models.sh llama-3-8b-q4ks

# Download all small models (<2GB)
./download_models.sh all-small

# Download all 7B/8B models
./download_models.sh all-7b
```

### Available Models

See [MODELS.md](MODELS.md) for a comprehensive list of available models, including:

- **Llama 2** (7B, 13B) - Base and Chat versions
- **Llama 3** (8B) - Latest generation with improved performance
- **Llama 3.1** (8B) - Enhanced capabilities
- **Llama 3.2** (1B, 3B) - Smaller, faster models

## Recommended Models

| Use Case | Model | Size | Best For |
|----------|-------|------|----------|
| Testing | Llama 3.2 1B (Q4_K_S) | 756 MB | Quick experiments |
| Development | Llama 3 8B (Q4_K_S) | 4.69 GB | Balanced performance |
| Production | Llama 3.1 8B (Q4_K_S) | 4.92 GB | Best quality |
| ARM Devices | Any Q4_0 model | Varies | Optimized speed |

## Usage with llama.cpp

After downloading models, run them with llama.cpp:

```bash
# Basic inference
./llama-cli -m models/Meta-Llama-3-8B-Instruct-Q4_K_S.gguf -p "Your prompt here"

# Interactive mode
./llama-cli -m models/llama-2-7b-chat.Q4_K_S.gguf -i

# With custom parameters
./llama-cli -m models/Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf \
  -n 512 \
  -t 8 \
  -p "Explain quantum computing"
```

## Repository Structure

```
.
├── README.md              # This file
├── MODELS.md             # Detailed model catalog
├── download_models.sh    # Download script
└── models/               # Downloaded model files
```

## Documentation

- [MODELS.md](MODELS.md) - Complete catalog of available models with download commands
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Inference engine documentation
- [Hugging Face](https://huggingface.co/models?other=GGUF) - Browse more GGUF models

## Requirements

- **wget** or **curl** for downloading
- **llama.cpp** for running models
- Sufficient disk space (models range from 750MB to 7GB)
- Recommended: 8GB+ RAM for 7B models, 16GB+ for 13B models

## License

The models are subject to their respective licenses from Meta AI and model creators. Please review:
- [Llama 2 License](https://ai.meta.com/llama/license/)
- [Llama 3 License](https://llama.meta.com/llama3/license/)

## Credits

- **Meta AI** - Original Llama models
- **TheBloke** - GGUF conversions of Llama 2 models
- **bartowski** - GGUF conversions of Llama 3/3.1/3.2 models
- **ggml.org** - llama.cpp inference engine
