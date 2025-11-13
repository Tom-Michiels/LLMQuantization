# CLAUDE.md - AI Assistant Guide for LLMQuantization

## Project Overview

**LLMQuantization** is a project focused on quantizing Large Language Models (LLMs) to reduce their memory footprint and computational requirements while maintaining acceptable performance levels.

### Purpose
- Implement various quantization techniques (INT8, INT4, mixed-precision, etc.)
- Provide tools and utilities for model compression
- Enable efficient deployment of LLMs on resource-constrained hardware
- Benchmark and compare different quantization methods

---

## Repository Structure

```
LLMQuantization/
├── README.md                  # Project introduction and quick start
├── CLAUDE.md                  # This file - AI assistant guidelines
├── src/                       # Source code (to be created)
│   ├── quantization/          # Core quantization algorithms
│   ├── models/                # Model wrappers and handlers
│   ├── benchmarks/            # Performance benchmarking tools
│   └── utils/                 # Utility functions
├── tests/                     # Unit and integration tests
├── experiments/               # Experimental notebooks and scripts
├── configs/                   # Configuration files for different models
├── docs/                      # Documentation
└── requirements.txt           # Python dependencies
```

**Current State**: Repository is in initial setup phase with minimal files.

---

## Development Workflow

### Branch Strategy
- **Main branch**: Stable, production-ready code
- **Feature branches**: Named as `claude/claude-md-*` or `feature/descriptive-name`
- **Current development branch**: `claude/claude-md-mhy05fdg538svo6q-01TcthhC8ZrqY2XYs3rjBuD5`

### Git Workflow
1. Always develop on the designated feature branch
2. Commit frequently with descriptive messages
3. Use conventional commit format:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `refactor:` for code restructuring
   - `test:` for test additions/changes
   - `docs:` for documentation updates
   - `perf:` for performance improvements
4. Push to origin using: `git push -u origin <branch-name>`

### Testing Requirements
- Write unit tests for all quantization algorithms
- Include integration tests for model pipelines
- Benchmark performance before and after optimizations
- Verify numerical accuracy within acceptable thresholds

---

## Key Technical Conventions

### Code Style
- **Language**: Python 3.8+
- **Formatting**: Follow PEP 8 guidelines
- **Type hints**: Use type annotations for function signatures
- **Docstrings**: Google-style docstrings for all public functions/classes

### Example Code Style
```python
from typing import Tuple, Optional
import torch

def quantize_weights(
    weights: torch.Tensor,
    bits: int = 8,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize model weights to specified bit precision.

    Args:
        weights: Input weight tensor to quantize
        bits: Number of bits for quantization (4, 8, or 16)
        symmetric: Whether to use symmetric quantization

    Returns:
        Tuple of (quantized_weights, scale_factors)

    Raises:
        ValueError: If bits not in [4, 8, 16]
    """
    # Implementation here
    pass
```

### Quantization-Specific Guidelines

1. **Precision Tracking**
   - Always track and report quantization error metrics
   - Use perplexity, accuracy loss, and inference speed as key metrics
   - Document numerical precision considerations

2. **Memory Management**
   - Be mindful of GPU memory usage during quantization
   - Implement gradient checkpointing where appropriate
   - Clear cache between large operations

3. **Model Compatibility**
   - Support common model architectures (GPT, LLaMA, BERT, etc.)
   - Handle different tensor formats (PyTorch, ONNX, etc.)
   - Provide fallback methods for unsupported operations

4. **Configuration Management**
   - Use YAML or JSON for quantization configs
   - Include presets for common use cases (mobile, edge, server)
   - Make hyperparameters easily adjustable

---

## Dependencies & Environment

### Core Dependencies (Expected)
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **NumPy**: Numerical computations
- **ONNX**: Model interchange format (optional)
- **bitsandbytes**: Quantization utilities (optional)

### Development Dependencies (Expected)
- **pytest**: Testing framework
- **black**: Code formatting
- **mypy**: Static type checking
- **jupyter**: Notebook experiments

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (when requirements.txt exists)
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

---

## AI Assistant Guidelines

### When Working on This Project

1. **Initial Setup Tasks**
   - Create proper directory structure
   - Set up requirements.txt with core dependencies
   - Initialize testing framework
   - Create basic configuration templates

2. **Implementing Quantization Features**
   - Start with well-documented reference implementations
   - Validate against known benchmarks (e.g., GPTQ, LLM.int8())
   - Include both calibration and inference code
   - Add comprehensive error handling

3. **Adding New Quantization Methods**
   - Document the theoretical basis in comments
   - Compare with baseline (FP16/FP32)
   - Provide example usage in docstring
   - Add corresponding test cases

4. **Performance Optimization**
   - Profile before optimizing
   - Document performance improvements with metrics
   - Consider both CPU and GPU implementations
   - Test on multiple hardware configurations if possible

5. **Code Review Checklist**
   - Type hints present and accurate
   - Docstrings complete and clear
   - Tests cover main functionality and edge cases
   - No hardcoded paths or credentials
   - Memory leaks prevented (proper cleanup)
   - Numerical stability verified

### Security Considerations
- Never commit model weights to the repository (use .gitignore)
- Don't hardcode API keys or credentials
- Validate all user inputs for file paths and parameters
- Be cautious with pickle files (potential security risk)

### Common Pitfalls to Avoid
- **Numerical instability**: Check for div-by-zero, overflow/underflow
- **Memory leaks**: Always detach tensors when not tracking gradients
- **Device mismatches**: Ensure tensors are on the correct device (CPU/GPU)
- **Shape mismatches**: Validate tensor dimensions before operations
- **Stale caching**: Clear CUDA cache when needed

---

## Testing Strategy

### Unit Tests
```python
# tests/test_quantization.py
import pytest
import torch
from src.quantization import quantize_weights

def test_int8_quantization():
    """Test 8-bit quantization produces expected range."""
    weights = torch.randn(100, 100)
    quant_weights, scales = quantize_weights(weights, bits=8)

    assert quant_weights.dtype == torch.int8
    assert quant_weights.min() >= -128
    assert quant_weights.max() <= 127
```

### Integration Tests
- Test full model quantization pipeline
- Verify inference produces reasonable outputs
- Check memory reduction meets targets

### Benchmarking
- Compare inference speed: FP32 vs FP16 vs INT8 vs INT4
- Measure model size reduction
- Track accuracy/perplexity degradation

---

## Documentation Standards

### Code Documentation
- All public functions/classes must have docstrings
- Include type hints for all parameters and returns
- Add usage examples for complex functions
- Document assumptions and limitations

### Repository Documentation
- **README.md**: High-level overview, installation, quick start
- **CLAUDE.md**: This file - detailed guidelines for AI assistants
- **docs/**: Detailed algorithm explanations, tutorials, API reference
- **CONTRIBUTING.md**: Guidelines for external contributors (when needed)

### Inline Comments
- Explain "why" not "what" (code should be self-documenting)
- Document complex mathematical operations
- Flag TODOs with context: `# TODO(context): description`

---

## Quantization-Specific Considerations

### Calibration Data
- Use representative dataset for calibration
- Document calibration dataset requirements
- Implement efficient data loading

### Quantization Modes
1. **Post-Training Quantization (PTQ)**: Quantize pre-trained models
2. **Quantization-Aware Training (QAT)**: Train with quantization in mind
3. **Dynamic Quantization**: Quantize activations at runtime
4. **Static Quantization**: Pre-compute quantization parameters

### Supported Quantization Schemes
- **Symmetric**: Zero-point at 0, equal positive/negative range
- **Asymmetric**: Arbitrary zero-point, optimized range
- **Per-tensor**: Single scale/zero-point for entire tensor
- **Per-channel**: Different scale/zero-point per output channel

---

## Current Development Phase

**Status**: Initial repository setup
**Priority Tasks**:
1. ✅ Create CLAUDE.md documentation
2. ⏳ Set up project structure (src/, tests/, configs/)
3. ⏳ Create requirements.txt with dependencies
4. ⏳ Implement basic INT8 quantization
5. ⏳ Add initial test suite
6. ⏳ Create example notebooks

---

## Resources & References

### Key Papers
- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- SmoothQuant: Accurate and Efficient Post-Training Quantization
- AWQ: Activation-aware Weight Quantization

### Useful Tools
- **Hugging Face Optimum**: Model optimization toolkit
- **ONNX Runtime**: Efficient inference engine
- **TensorRT**: NVIDIA's optimization library
- **Intel Neural Compressor**: Quantization and optimization

### Community Resources
- [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html)
- [Hugging Face Model Optimization](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one)

---

## Notes for AI Assistants

### When Asked to Implement Features
1. Always check existing code structure first
2. Follow the established patterns and conventions
3. Write tests alongside implementation
4. Update documentation as you code
5. Use TodoWrite to track multi-step tasks

### When Debugging
1. Check tensor shapes and dtypes first
2. Verify device placement (CPU vs GPU)
3. Look for numerical issues (NaN, Inf)
4. Profile memory usage for large models
5. Validate against reference implementations

### When Optimizing
1. Establish baseline metrics first
2. Profile to find bottlenecks
3. Optimize the slowest parts first
4. Verify correctness after optimization
5. Document performance improvements

---

## Version History

- **2025-11-13**: Initial CLAUDE.md creation
  - Established project structure and guidelines
  - Defined coding conventions and workflows
  - Added quantization-specific best practices

---

## Contact & Contribution

This is an active development project. When contributing:
- Follow the conventions outlined in this document
- Write clear commit messages
- Include tests for new features
- Update documentation as needed
- Push to appropriate feature branches

For questions or clarifications, refer to the main README.md or project documentation.
