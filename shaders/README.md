# Example Compute Shaders

This directory contains example GLSL compute shaders for common GPU operations.

## Shaders

### Core Operations

| Shader | Description | Bindings |
|--------|-------------|----------|
| `vector_add.comp` | Element-wise vector addition (C = A + B) | 3 storage buffers |
| `vector_scale.comp` | Vector scaling (B = A * scale) | 2 storage buffers |
| `saxpy.comp` | BLAS Level 1 SAXPY (Y = alpha*X + Y) | 2 storage buffers |
| `reduce_sum.comp` | Parallel reduction sum | 2 storage buffers |
| `matrix_multiply.comp` | Tiled matrix multiplication (C = A*B) | 3 storage buffers |

### ML / Neural Network Operations

| Shader | Description | Bindings |
|--------|-------------|----------|
| `relu.comp.glsl` | ReLU activation function | 2 storage buffers |
| `softmax.comp.glsl` | Softmax activation (per-row) | 2 storage buffers |
| `conv2d.comp.glsl` | 2D convolution (3x3 kernel) | 3 storage buffers |

### Advanced Algorithms

| Shader | Description | Bindings |
|--------|-------------|----------|
| `reduction.comp.glsl` | Parallel sum reduction with shared memory | 2 storage buffers |
| `prefix_sum.comp.glsl` | Parallel prefix sum (Blelloch algorithm) | 1 storage buffer |

## Compilation

### Prerequisites

Install a GLSL to SPIR-V compiler:

```bash
# Ubuntu/Debian
sudo apt install glslang-tools

# Fedora
sudo dnf install glslang

# Arch
sudo pacman -S glslang

# macOS (with Homebrew)
brew install glslang
```

### Compile Shaders

```bash
# Make the script executable
chmod +x compile.sh

# Compile all shaders
./compile.sh

# Clean compiled files
./compile.sh clean
```

### Manual Compilation

```bash
# Using glslangValidator
glslangValidator -V --target-env vulkan1.0 -o vector_add.spv vector_add.comp

# Using shaderc (glslc)
glslc --target-env=vulkan1.0 -O -o vector_add.spv vector_add.comp
```

## Usage in Zig

### Embedded SPIR-V

```zig
const vulkan = @import("vulkan");

// Embed compiled SPIR-V at compile time
const shader_code = @embedFile("shaders/vector_add.spv");

// Load shader module
var shader = try vulkan.ShaderModule.initFromBytes(&ctx, shader_code);
defer shader.deinit();
```

### Runtime Loading

```zig
// Load from file at runtime
var shader = try vulkan.ShaderModule.initFromFile(&ctx, "shaders/vector_add.spv");
defer shader.deinit();
```

## Shader Details

### Vector Add (`vector_add.comp`)

Performs element-wise addition of two vectors.

**Inputs:**
- Binding 0: Input vector A (float[])
- Binding 1: Input vector B (float[])
- Push constant: `n` (element count)

**Output:**
- Binding 2: Result vector C (float[])

**Dispatch:** `ceil(n / 256)` workgroups

### Vector Scale (`vector_scale.comp`)

Multiplies each element of a vector by a scalar.

**Inputs:**
- Binding 0: Input vector (float[])
- Push constant: `{ scale, n }`

**Output:**
- Binding 1: Result vector (float[])

**Dispatch:** `ceil(n / 256)` workgroups

### SAXPY (`saxpy.comp`)

Computes the BLAS Level 1 operation: Y = alpha * X + Y

**Inputs:**
- Binding 0: Input vector X (float[])
- Binding 1: Input/Output vector Y (float[])
- Push constant: `{ alpha, n }`

**Dispatch:** `ceil(n / 256)` workgroups

### Reduce Sum (`reduce_sum.comp`)

Parallel reduction to compute sum of array elements.

**Inputs:**
- Binding 0: Input array (float[])
- Push constant: `n` (element count)

**Output:**
- Binding 1: Partial sums (one per workgroup)

**Dispatch:** `ceil(n / 512)` workgroups

*Note: For arrays larger than 512 elements, multiple reduction passes are needed.*

### Matrix Multiply (`matrix_multiply.comp`)

Tiled matrix multiplication using shared memory for better cache utilization.

**Inputs:**
- Binding 0: Matrix A (M x K, row-major)
- Binding 1: Matrix B (K x N, row-major)
- Push constant: `{ M, N, K }`

**Output:**
- Binding 2: Matrix C (M x N, row-major)

**Dispatch:** `ceil(N / 16)` x `ceil(M / 16)` workgroups

## Performance Tips

1. **Workgroup Size**: The shaders use 256 threads per workgroup (16x16 for matrix multiply), which is optimal for most GPUs.

2. **Memory Coalescing**: Access patterns are designed for coalesced memory access when possible.

3. **Shared Memory**: `reduce_sum.comp` and `matrix_multiply.comp` use shared memory to reduce global memory bandwidth.

4. **Occupancy**: Keep input sizes as multiples of workgroup size for best GPU utilization.
