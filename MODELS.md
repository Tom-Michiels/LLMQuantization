# Popular Llama Models with Q4_0, Q4_1, and Q4_K_S Quantization

This document lists popular Llama models available in GGUF format with Q4_0, Q4_1, and Q4_K_S quantization formats.

## About Quantization Types

### Q4_0
- 4-bit quantization using type 0 method (older)
- Produces smaller models but may have slightly higher perplexity
- Optimized for ARM chips with substantial speedup
- ~3.5GB for 7B models

### Q4_1
- 4-bit quantization using type 1 method
- Better quality than Q4_0 with similar size
- ~4GB for 7B models

### Q4_K_S (Small)
- Part of the K-quant family (llama.cpp's newer quantization)
- Uses "type-1" 4-bit quantization in super-blocks
- 4.5 bits per weight (bpw)
- Better quality than Q4_0/Q4_1 with similar size
- ~3.56GB for 7B models

## Popular Models

### 1. Llama 2 Models (TheBloke)

#### Llama-2-7B-GGUF
- **Repository**: TheBloke/Llama-2-7B-GGUF
- **Base Model**: Meta's Llama 2 7B
- **Available Quantizations**: Q4_0, Q4_K_S, Q4_K_M
- **Download Commands**:
  ```bash
  # Q4_K_S (3.56 GB)
  wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_S.gguf

  # Q4_0 (3.50 GB)
  wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf
  ```

#### Llama-2-7B-Chat-GGUF
- **Repository**: TheBloke/Llama-2-7B-Chat-GGUF
- **Base Model**: Meta's Llama 2 7B Chat (instruction-tuned)
- **Available Quantizations**: Q4_0, Q4_K_S, Q4_K_M
- **Download Commands**:
  ```bash
  # Q4_K_S (3.56 GB)
  wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_S.gguf

  # Q4_0 (3.50 GB)
  wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf
  ```

#### Llama-2-13B-GGUF
- **Repository**: TheBloke/Llama-2-13B-GGUF
- **Base Model**: Meta's Llama 2 13B
- **Available Quantizations**: Q4_0, Q4_K_S, Q4_K_M
- **Download Commands**:
  ```bash
  # Q4_K_S (7.16 GB)
  wget https://huggingface.co/TheBloke/Llama-2-13B-GGUF/resolve/main/llama-2-13b.Q4_K_S.gguf

  # Q4_0 (6.86 GB)
  wget https://huggingface.co/TheBloke/Llama-2-13B-GGUF/resolve/main/llama-2-13b.Q4_0.gguf
  ```

### 2. Llama 3 Models (bartowski)

#### Meta-Llama-3-8B-Instruct-GGUF
- **Repository**: bartowski/Meta-Llama-3-8B-Instruct-GGUF
- **Base Model**: Meta's Llama 3 8B Instruct
- **Available Quantizations**: Q4_0, Q4_K_S, Q4_K_M
- **Download Commands**:
  ```bash
  # Q4_K_S (4.69 GB)
  wget https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_S.gguf

  # Q4_0 (4.33 GB)
  wget https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_0.gguf
  ```

#### Meta-Llama-3.1-8B-Instruct-GGUF
- **Repository**: bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
- **Base Model**: Meta's Llama 3.1 8B Instruct
- **Available Quantizations**: Q4_0, Q4_K_S, Q4_K_M
- **Download Commands**:
  ```bash
  # Q4_K_S (4.92 GB)
  wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf

  # Q4_0 (4.66 GB)
  wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_0.gguf
  ```

#### Llama-3.2-3B-Instruct-GGUF
- **Repository**: bartowski/Llama-3.2-3B-Instruct-GGUF
- **Base Model**: Meta's Llama 3.2 3B Instruct (smaller, faster)
- **Available Quantizations**: Q4_K_S, Q4_K_M
- **Download Commands**:
  ```bash
  # Q4_K_S (1.95 GB)
  wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_S.gguf
  ```

#### Llama-3.2-1B-Instruct-GGUF
- **Repository**: bartowski/Llama-3.2-1B-Instruct-GGUF
- **Base Model**: Meta's Llama 3.2 1B Instruct (smallest, fastest)
- **Available Quantizations**: Q4_K_S, Q4_K_M, Q8_0
- **Download Commands**:
  ```bash
  # Q4_K_S (756 MB)
  wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_S.gguf
  ```

### 3. LLaMA Pro Models (TheBloke)

#### LLaMA-Pro-8B-Instruct-GGUF
- **Repository**: TheBloke/LLaMA-Pro-8B-Instruct-GGUF
- **Base Model**: LLaMA Pro 8B Instruct (enhanced version)
- **Available Quantizations**: Q4_0, Q4_K_S, Q4_K_M
- **Download Commands**:
  ```bash
  # Q4_K_S
  wget https://huggingface.co/TheBloke/LLaMA-Pro-8B-Instruct-GGUF/resolve/main/llama-pro-8b-instruct.Q4_K_S.gguf

  # Q4_0
  wget https://huggingface.co/TheBloke/LLaMA-Pro-8B-Instruct-GGUF/resolve/main/llama-pro-8b-instruct.Q4_0.gguf
  ```

## Alternative Download Methods

### Using Hugging Face CLI
If you have `huggingface-cli` installed:

```bash
# Install huggingface-cli
pip install -U huggingface-hub

# Download specific quantization
huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_S.gguf --local-dir ./models --local-dir-use-symlinks False

# Download all Q4_K_S files from a repo
huggingface-cli download TheBloke/Llama-2-7B-GGUF --local-dir ./models --local-dir-use-symlinks False --include='*Q4_K_S*.gguf'
```

### Using Git LFS
```bash
# Clone the repository (warning: may be very large)
git lfs install
git clone https://huggingface.co/TheBloke/Llama-2-7B-GGUF

# Or use sparse checkout for specific files
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/TheBloke/Llama-2-7B-GGUF
cd Llama-2-7B-GGUF
git lfs pull --include="llama-2-7b.Q4_K_S.gguf"
```

## Recommended Models by Use Case

### For General Testing (Small & Fast)
- **Llama-3.2-1B-Instruct** (Q4_K_S) - 756 MB
- **Llama-3.2-3B-Instruct** (Q4_K_S) - 1.95 GB

### For Development & Experimentation (Balanced)
- **Meta-Llama-3-8B-Instruct** (Q4_K_S) - 4.69 GB
- **Llama-2-7B-Chat** (Q4_K_S) - 3.56 GB

### For Better Quality (Larger)
- **Meta-Llama-3.1-8B-Instruct** (Q4_K_S) - 4.92 GB
- **Llama-2-13B** (Q4_K_S) - 7.16 GB

### For ARM/Mobile Devices
- Any model with **Q4_0** quantization for optimal ARM performance

## Performance Comparison

| Model | Quantization | Size | Speed | Quality |
|-------|-------------|------|-------|---------|
| Llama-3.2-1B | Q4_K_S | 756 MB | Fastest | Good for simple tasks |
| Llama-2-7B | Q4_0 | 3.50 GB | Very Fast | Good |
| Llama-2-7B | Q4_K_S | 3.56 GB | Fast | Better |
| Llama-3-8B | Q4_K_S | 4.69 GB | Fast | Excellent |
| Llama-3.1-8B | Q4_K_S | 4.92 GB | Fast | Best |
| Llama-2-13B | Q4_K_S | 7.16 GB | Moderate | Excellent |

## Usage with llama.cpp

After downloading, you can run these models with llama.cpp:

```bash
# Basic usage
./llama-cli -m models/llama-2-7b.Q4_K_S.gguf -p "Your prompt here"

# Interactive mode
./llama-cli -m models/Meta-Llama-3-8B-Instruct-Q4_K_S.gguf -i

# With specific parameters
./llama-cli -m models/llama-2-7b-chat.Q4_K_S.gguf \
  -n 512 \
  -t 8 \
  -p "Your prompt here"
```

## Additional Resources

- **Hugging Face Collections**: Search for "GGUF" on Hugging Face
- **llama.cpp Documentation**: https://github.com/ggml-org/llama.cpp
- **Quantization Guide**: https://github.com/ggml-org/llama.cpp/blob/master/examples/quantize/README.md
