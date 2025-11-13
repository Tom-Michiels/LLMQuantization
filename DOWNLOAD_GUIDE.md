# Download Guide for Llama Quantized Models

This guide provides multiple methods to download the popular Llama models with Q4_0, Q4_1, and Q4_K_S quantization.

## Network Requirements

**Important**: Downloading these models requires:
- Unrestricted internet access to Hugging Face (huggingface.co)
- No firewall/proxy blocking (no 403 Forbidden errors)
- Sufficient bandwidth and disk space

## Method 1: Python Script (Recommended)

Use the provided `download_models.py` script:

```bash
# Download a single model
python3 download_models.py llama-3.2-1b-q4ks

# Download all small models (<2GB)
python3 download_models.py all-small

# Download all 7B/8B models
python3 download_models.py all-7b

# Download everything
python3 download_models.py all
```

### Available model keys:
- `llama-3.2-1b-q4ks` - Llama 3.2 1B (756 MB)
- `llama-3.2-3b-q4ks` - Llama 3.2 3B (1.95 GB)
- `llama-2-7b-q4ks` - Llama 2 7B (3.56 GB)
- `llama-2-7b-chat-q4ks` - Llama 2 7B Chat (3.56 GB)
- `llama-2-7b-q40` - Llama 2 7B Q4_0 (3.50 GB)
- `llama-3-8b-q4ks` - Llama 3 8B (4.69 GB)
- `llama-3-8b-q40` - Llama 3 8B Q4_0 (4.33 GB)
- `llama-3.1-8b-q4ks` - Llama 3.1 8B (4.92 GB)
- `llama-2-13b-q4ks` - Llama 2 13B (7.16 GB)

## Method 2: Bash Script

Use the `download_models.sh` script (requires wget):

```bash
./download_models.sh llama-3-8b-q4ks
```

## Method 3: Manual wget Downloads

Download directly using wget:

```bash
cd models

# Small models - great for testing
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_S.gguf

wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_S.gguf

# Medium models - balanced performance
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_S.gguf

wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_S.gguf

wget https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_S.gguf

# Latest models - best quality
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf

# Q4_0 variants - optimized for ARM
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf

wget https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_0.gguf

# Large model
wget https://huggingface.co/TheBloke/Llama-2-13B-GGUF/resolve/main/llama-2-13b.Q4_K_S.gguf
```

## Method 4: Using Hugging Face CLI

Install and use the official CLI:

```bash
# Install
pip install -U huggingface-hub

# Download models
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_S.gguf --local-dir ./models --local-dir-use-symlinks False

huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_S.gguf --local-dir ./models --local-dir-use-symlinks False

huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_S.gguf --local-dir ./models --local-dir-use-symlinks False

huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_S.gguf --local-dir ./models --local-dir-use-symlinks False

huggingface-cli download bartowski/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_S.gguf --local-dir ./models --local-dir-use-symlinks False

huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf --local-dir ./models --local-dir-use-symlinks False
```

## Method 5: Using curl

Alternative to wget:

```bash
cd models

curl -L -o Llama-3.2-1B-Instruct-Q4_K_S.gguf \
  "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_S.gguf"

curl -L -o Meta-Llama-3-8B-Instruct-Q4_K_S.gguf \
  "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_S.gguf"
```

## Recommended Download Order

### For Quick Testing:
1. `llama-3.2-1b-q4ks` (756 MB) - Smallest, fastest download
2. `llama-3.2-3b-q4ks` (1.95 GB) - Still small, better quality

### For Development:
1. `llama-2-7b-chat-q4ks` (3.56 GB) - Proven, chat-optimized
2. `llama-3-8b-q4ks` (4.69 GB) - Latest generation

### For Production/Best Quality:
1. `llama-3.1-8b-q4ks` (4.92 GB) - Latest and best 8B model
2. `llama-2-13b-q4ks` (7.16 GB) - Larger, more capable

### For ARM Devices:
- Download Q4_0 variants instead of Q4_K_S for better performance

## Disk Space Requirements

| Models | Total Size |
|--------|------------|
| All small (<2GB) | ~2.7 GB |
| All 7B/8B Q4_K_S | ~16.7 GB |
| All available models | ~37 GB |

## Troubleshooting

### 403 Forbidden Error
- Check if you're behind a firewall/proxy
- Try using a VPN
- Ensure Hugging Face isn't blocked in your network
- Try downloading from a different network

### Slow Downloads
- Use `-c` flag with wget to resume interrupted downloads
- Try curl instead of wget
- Download during off-peak hours
- Consider downloading smaller models first

### Disk Space Issues
- Check available space: `df -h`
- Download models one at a time
- Remove unnecessary files
- Consider downloading only Q4_K_S OR Q4_0, not both

### Verification
After downloading, verify file sizes:

```bash
ls -lh models/

# Expected sizes (approximate):
# Llama 3.2 1B:  756 MB
# Llama 3.2 3B:  1.95 GB
# Llama 2 7B:    3.56 GB
# Llama 3 8B:    4.69 GB
# Llama 3.1 8B:  4.92 GB
# Llama 2 13B:   7.16 GB
```

## Using Downloaded Models

Once downloaded, test with llama.cpp:

```bash
# Basic test
./llama-cli -m models/Llama-3.2-1B-Instruct-Q4_K_S.gguf -p "Hello, who are you?" -n 100

# Interactive mode
./llama-cli -m models/Meta-Llama-3-8B-Instruct-Q4_K_S.gguf -i

# With system prompt
./llama-cli -m models/llama-2-7b-chat.Q4_K_S.gguf \
  --system "You are a helpful coding assistant" \
  -p "Explain recursion in Python"
```

## Additional Resources

- [MODELS.md](MODELS.md) - Complete model catalog
- [Hugging Face GGUF Models](https://huggingface.co/models?other=GGUF)
- [llama.cpp Documentation](https://github.com/ggml-org/llama.cpp)
- [Quantization Guide](https://github.com/ggml-org/llama.cpp/blob/master/examples/quantize/README.md)
