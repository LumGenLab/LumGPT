# GPT C++ Implementation from Scratch

A complete implementation of a GPT-style transformer language model written in C++ from scratch, featuring multi-head attention, layer normalization, feed-forward networks, and AdamW optimization.

## Quick Start

1. Open Command Prompt (CMD)
2. Navigate to the folder containing `lum_gpt.cpp`
3. Paste this command to compile:
   ```bash
   g++ -std=c++17 -O3 -march=native lum_gpt.cpp -o gpt_model.exe
   ```
4. You can add more flags like `-ffast-math` or debugging flags if you modified the code and added more features and want to debug the code.
5. After several seconds, it will compile into `gpt_model.exe`
6. Just double click `gpt_model.exe` to run!

The program automatically downloads the TinyShakespeare dataset if not present. If the library isn't present which downloads the dataset so it is then recommended to download it manually.

## Hardware Performance

**Test System:**
- CPU: AMD Phenom™ Triple-Core Processor @ 2.40 GHz
- RAM: 2 GB DDR2 (700 MB available)
- Storage: 149 GB HDD
- GPU: None (GTX 210 only for display)

**Resource Usage During Training:**
- Memory: 32 MB
- CPU: 45%
- Disk: 0-2%
- Training Time: ~8 minutes per 200 iterations

## Model Architecture

Complete transformer implementation with:
- Multi-head attention with causal masking
- Layer normalization (Pre-LN as in GPT-2/3)
- Feed-forward networks with GELU activation
- AdamW optimizer with decoupled weight decay
- Advanced text generation (Temperature + Top-K sampling)

## Training Results

### TinyShakespeare Dataset
- **Characters**: 1.1M
- **Vocabulary**: 65 unique characters
- **Lines**: ~40,000

**Original Hyperparameters (output.txt):**
```cpp
batch_size = 4, block_size = 64, d_model = 128, n_heads = 4, n_layers = 4
```

**Loss Progress:**
```
Step 0: 4.5875
Step 200: 3.1597
Step 400: 3.1563
...
Step 2000: 3.2377
```

### Custom Nasiruddin Dataset
- **Content**: 202 jokes from Internet Archive
- **Vocabulary**: 82 unique characters  
- **Lines**: ~3,000
- **Quality**: More modern, clearer English than TinyShakespeare
- **Availability**: The custom dataset is also added to the repository so you can use. Simply, change the path of the dataset in the code.

**Enhanced Hyperparameters:**
```cpp
batch_size = 6, block_size = 128, d_model = 256, n_heads = 6, n_layers = 6
```

## Technical Implementation

### Core Components
1. **Tensor Operations**: Custom tensor class with optimized matrix operations
2. **Embeddings**: Token and positional embeddings with Xavier initialization
3. **Multi-Head Attention**: Scaled dot-product attention implementation
4. **Layer Normalization**: Mathematically precise gradient computation
5. **Feed-Forward Networks**: MLP with GELU activation
6. **AdamW Optimizer**: Adam with decoupled weight decay

### Mathematical Precision
- All gradients computed using exact mathematical derivations
- Numerical stability through epsilon constants and max trick
- Proper gradient accumulation and backpropagation
- Combined softmax-cross entropy gradients for efficiency

### Memory Optimization
- Flattened tensor storage for cache efficiency
- Thread-local random number generation
- Careful buffer management and reuse
- In-place operations where possible

## Production Grade Features

This is **not just educational** - it's a **full production-grade implementation** suitable for real applications. The code demonstrates:
- Industry-standard mathematical implementations
- Memory-efficient design for resource-constrained environments
- Robust numerical stability
- Modular, maintainable architecture

## Next Version (Coming Soon)

The next version will include cutting-edge optimizations:
- **4-bit Quantization QAT** (Quantization Aware Training)
- **RoPE** (Rotary Position Embedding)
- **ALiBi** (Attention with Linear Biases)
- **Eigen 3.4.0** integration for ultra-optimized linear algebra
- **Custom inference engine** with specialized optimizations
- **Ultra-efficient memory management**
- **Production inference support**

Both versions are designed for production use with educational value as a bonus.

## Key Achievements

1. **First attempt** at transformer implementation from scratch
2. **Runs on 15+ year old hardware** with excellent performance
3. **Complete mathematical implementation** with proper gradients
4. **Production-ready code quality** with comprehensive error handling
5. **Custom dataset compatibility** with automatic vocabulary building

## Model Configuration

Current implementation supports:
- Variable vocabulary sizes (62-82+ characters tested)
- Adjustable context windows (64-128+ tokens)
- Scalable model dimensions (128-256+ features)
- Flexible batch processing
- Custom dataset integration

## Technical Specifications

| Component | Implementation |
|-----------|----------------|
| Language | C++17 |
| Dependencies | Standard library only |
| Memory Model | Flattened tensors |
| Optimization | AdamW with weight decay |
| Attention | Multi-head with causal mask |
| Generation | Temperature + Top-K sampling |
| Dataset | Auto-download capability |

## Code Quality

- **2,000+ lines** of well-documented C++
- **Mathematical comments** explaining derivations
- **Error handling** and numerical stability
- **Modular design** with clear separation
- **Performance optimizations** throughout

## Future Development

This project represents the foundation for advanced transformer research and development. The upcoming version will push the boundaries of efficient transformer implementation while maintaining the educational clarity of the current version. This version doesn't include the inference support but it will be added in the upcoming version which will allow to train the model once and use the weights and vocabulary files for inference.

**Note:** No contributions accepted - this is a personal research project.

---

*A complete transformer implementation proving that deep learning doesn't require expensive hardware or complex frameworks.*
