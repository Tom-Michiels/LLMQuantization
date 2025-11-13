# Usage Guide

Due to network restrictions in some environments, you may need to download the models manually or run the scripts in an environment with proper HuggingFace access.

## Option 1: Automated Download (Requires HuggingFace Access)

```bash
# Install dependencies
pip install -r requirements.txt

# Download all models
python3 download_models.py
```

## Option 2: Manual Download

If automated download fails, download the models manually from HuggingFace:

1. Visit: https://huggingface.co/QuantFactory/Llama-3.2-1B-GGUF
2. Download these files to the `models/` directory:
   - `Llama-3.2-1B.Q4_0.gguf` (~771 MB)
   - `Llama-3.2-1B.Q4_1.gguf` (~832 MB)
   - `Llama-3.2-1B.Q4_K_S.gguf` (~776 MB)

Alternatively, use `huggingface-cli`:

```bash
huggingface-cli download QuantFactory/Llama-3.2-1B-GGUF \
  Llama-3.2-1B.Q4_0.gguf \
  Llama-3.2-1B.Q4_1.gguf \
  Llama-3.2-1B.Q4_K_S.gguf \
  --local-dir models/
```

Or use `wget` (if proxy/auth allows):

```bash
mkdir -p models
cd models

wget https://huggingface.co/QuantFactory/Llama-3.2-1B-GGUF/resolve/main/Llama-3.2-1B.Q4_0.gguf
wget https://huggingface.co/QuantFactory/Llama-3.2-1B-GGUF/resolve/main/Llama-3.2-1B.Q4_1.gguf
wget https://huggingface.co/QuantFactory/Llama-3.2-1B-GGUF/resolve/main/Llama-3.2-1B.Q4_K_S.gguf
```

## Running the Full Experiment

Once models are downloaded:

```bash
# Run the complete experiment
bash run_experiment.sh
```

This will:
1. Setup llama.cpp and WikiText-2
2. Measure baseline perplexity for all models
3. Create modified versions (RMSNorm fp16)
4. Re-measure perplexity for modified models
5. Generate comparison report

## Step-by-Step Manual Execution

### 1. Setup llama.cpp
```bash
bash setup_llamacpp.sh
```

### 2. Measure Baseline Perplexity
```bash
# For each model
./llama.cpp/perplexity -m models/Llama-3.2-1B.Q4_0.gguf -f wikitext-2/wiki.test.raw -ngl 0
./llama.cpp/perplexity -m models/Llama-3.2-1B.Q4_1.gguf -f wikitext-2/wiki.test.raw -ngl 0
./llama.cpp/perplexity -m models/Llama-3.2-1B.Q4_K_S.gguf -f wikitext-2/wiki.test.raw -ngl 0
```

### 3. Modify Models
```bash
python3 modify_rmsnorm.py
```

This creates modified versions in `models/modified/` with RMSNorm weights rounded to fp16.

### 4. Measure Modified Model Perplexity
```bash
# For each modified model
./llama.cpp/perplexity -m models/modified/modified_Llama-3.2-1B.Q4_0.gguf -f wikitext-2/wiki.test.raw -ngl 0
./llama.cpp/perplexity -m models/modified/modified_Llama-3.2-1B.Q4_1.gguf -f wikitext-2/wiki.test.raw -ngl 0
./llama.cpp/perplexity -m models/modified/modified_Llama-3.2-1B.Q4_K_S.gguf -f wikitext-2/wiki.test.raw -ngl 0
```

### 5. Generate Comparison Report
```bash
python3 measure_perplexity.py
```

## Expected Output

The experiment will generate:
- `results/perplexity_results.json`: Complete perplexity measurements
- `results/baseline_perplexity.log`: Raw output from baseline measurements
- `results/modified_perplexity.log`: Raw output from modified model measurements
- Console output showing comparison table

## Troubleshooting

### Download Issues
- **403 Forbidden**: May need HuggingFace token for some repos
- **Network errors**: Try using a VPN or different network
- **Slow downloads**: Models are ~800MB each, may take time

### Build Issues
- Ensure you have `make` and `gcc`/`clang` installed
- llama.cpp requires C++11 compiler
- For GPU support, see llama.cpp documentation

### Memory Issues
- Perplexity measurement requires ~2-4GB RAM
- Use `-ngl 0` to force CPU mode (default in scripts)
- Consider using smaller context with `-c` flag if needed

## Understanding Results

Results will show perplexity for each model:
- **Lower perplexity** = better language modeling
- **Difference** = Impact of fp16 RMSNorm weights
- Typically expect small increase (~0.1-0.5) in perplexity with fp16

Example output:
```
Model                         Original        Modified (fp16)  Difference
--------------------------------------------------  -------------------------
Llama-3.2-1B.Q4_0.gguf       12.3456         12.4123         +0.0667
Llama-3.2-1B.Q4_1.gguf       12.2345         12.3012         +0.0667
Llama-3.2-1B.Q4_K_S.gguf     12.1234         12.1901         +0.0667
```
