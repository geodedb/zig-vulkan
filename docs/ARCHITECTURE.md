# Architecture

This document describes the architecture of the Zig Vulkan Compute Library (v0.6.0).

## Overview

The library is organized into four main layers:

```
┌────────────────────────────────────────────────────────────────────┐
│                          Application                                │
├────────────────────────────────────────────────────────────────────┤
│                      High-Level API                                 │
│  Context, Buffer, Pipeline, CommandBuffer, ComputeGraph, etc.     │
├────────────────────────────────────────────────────────────────────┤
│                    Feature Modules                                  │
│  Extensions, Multi-Queue, Profiling, Recovery, Hot-Reload, etc.   │
├────────────────────────────────────────────────────────────────────┤
│                    Low-Level Bindings                              │
│  types.zig, loader.zig, error.zig                                  │
├────────────────────────────────────────────────────────────────────┤
│                      Vulkan Loader                                  │
│  Dynamic library loading, function pointers                        │
├────────────────────────────────────────────────────────────────────┤
│                    Vulkan Driver/ICD                               │
└────────────────────────────────────────────────────────────────────┘
```

## Module Structure

### `src/vk/` - Low-Level Vulkan Bindings

**`types.zig`**
- All Vulkan type definitions
- Enumerations and constants
- Structure definitions matching Vulkan specification
- Zero-cost abstractions where possible
- Descriptor update template types

**`loader.zig`**
- Dynamic library loading (`libvulkan.so.1`, `vulkan-1.dll`, etc.)
- Function pointer dispatch tables:
  - `BaseLoader`: Initial library load and vkGetInstanceProcAddr
  - `GlobalDispatch`: Pre-instance functions
  - `InstanceDispatch`: Instance-level functions
  - `DeviceDispatch`: Device-level functions
- Platform-specific library name handling

**`error.zig`**
- Unified error handling with VkResult to Zig error mapping
- `ErrorContext`: Detailed error information with source location
- `ErrorTrace`: Error chaining for debugging
- `ErrorLog`: Global error tracking
- `Diagnostics`: Error suggestions and related errors

### `src/compute/` - High-Level Compute API

**`context.zig`**
- Main entry point for the library
- `Context`: Manages Vulkan instance, device, queues
- `Options`: Configuration for context creation
- `ProbeResult`: System capability detection
- `DeviceInfo`: Physical device information
- Debug utilities integration

**`buffer.zig`**
- `Buffer`: GPU buffer abstraction
- `BufferUsage`: Semantic buffer usage patterns (storage, staging, readback, uniform)
- `Transfer`: Buffer copy and barrier operations
- `PooledBuffer`: Buffer with suballocation support
- Memory management integration

**`pipeline.zig`**
- `ShaderModule`: SPIR-V shader loading with validation
- `DescriptorSetLayout`: Descriptor binding specifications
- `DescriptorPool`: Descriptor set allocation
- `ComputePipeline`: Compute pipeline management
- `PipelineCache`: Pipeline compilation caching
- `PersistentPipelineCache`: Disk-backed pipeline cache with auto-save
- Specialization constants support

**`command.zig`**
- `CommandBuffer`: Command recording abstraction
- `Fence`: CPU-GPU synchronization
- `Semaphore`: GPU-GPU synchronization
- `Submission`: Queue submission helpers
- `TimestampQuery`: Performance measurement
- `PipelineStatisticsQuery`: GPU workload statistics

**`queue.zig`**
- `QueueManager`: Multi-queue management
- `QueueFamilyInfo`: Queue family enumeration
- `TimelineSemaphore`: Timeline-based synchronization
- `QueueSync`: Cross-queue synchronization barriers
- `CrossQueueCoordinator`: High-level queue coordination
- `BufferOwnershipTransfer`, `ImageOwnershipTransfer`

**`dispatch.zig`**
- `ComputeDispatcher`: High-level dispatch helper
- `WorkgroupSize`: Workgroup configuration
- `DispatchDimensions`: Automatic workgroup calculation
- `DispatchBatch`: Batched dispatch recording

**`multi_device.zig`**
- `MultiDeviceContext`: Multi-GPU management
- `PhysicalDeviceHandle`: Per-device wrapper
- `DeviceSelector`: Device selection criteria
- `DeviceCapabilities`: Device feature detection
- `DevicePool`: Device load balancing

**`extensions.zig`**
- Extension capability queries:
  - `SubgroupCapabilities`: Subgroup operations
  - `DescriptorIndexingCapabilities`: Bindless descriptors
  - `RayTracingCapabilities`: Ray tracing support
  - `VideoCapabilities`: Video encode/decode
  - `CooperativeMatrixCapabilities`: Matrix operations for ML
  - `SparseResourceCapabilities`: Sparse binding
  - `MeshShaderCapabilities`: Mesh shaders
  - `PushDescriptorCapabilities`: Push descriptors
- `MemoryBudgetInfo`: Real-time memory budget tracking
- `MemoryBudgetMonitor`: Memory status monitoring
- `MemoryPressureHandler`: Automatic resource eviction

**`async_compute.zig`**
- `AsyncCoordinator`: Overlapping compute/transfer
- `FrameSync`: Frame-based synchronization
- `StreamBuffer`: Double-buffer helpers
- Resource state tracking

**`graph.zig`**
- `ComputeGraph`: DAG-based task scheduling
- `TaskNode`: Compute task definition
- `ResourceHandle`, `ResourceUsage`: Resource tracking
- Automatic barrier insertion
- Parallel task execution by levels

**`profiling.zig`**
- `ComputeProfiler`: GPU performance profiling
- `ProfileResult`: Timing measurements
- `ProfileStats`: Aggregated statistics
- `ProfileScope`: Scoped profiling helpers

### `src/util/` - Utilities

**`allocator.zig`**
- `SimpleAllocator`: Basic GPU memory allocator
- `MemoryUsage`: Memory usage patterns
- `Alignment`: Alignment utilities

**`pooled_allocator.zig`**
- `PooledAllocator`: Efficient memory suballocation
- `PoolConfig`: Pool configuration
- `Suballocation`: Allocation result with offset
- `AllocatorStats`: Fragmentation tracking

**`validation.zig`**
- SPIR-V validation utilities
- `SpirvValidationResult`: Validation status
- `SpirvInfo`: Shader metadata extraction

**`glsl.zig`**
- `Compiler`: GLSL to SPIR-V compilation (glslangValidator/glslc)
- `HlslCompiler`: HLSL to SPIR-V compilation (DXC)
- `UnifiedCompiler`: Multi-language support
- `isCompilerAvailable()`, `isHlslCompilerAvailable()`
- `detectLanguage()`: Automatic source language detection

**`reflection.zig`**
- `ShaderReflection`: SPIR-V reflection
- `ReflectedBinding`: Descriptor binding info
- `PushConstantInfo`: Push constant layout

**`error_recovery.zig`**
- `DeviceLostHandler`: Automatic device recovery
- `RecreationError`: Recovery error types
- `DeviceLostConfig`: Recovery configuration
- `DeviceLostStats`: Recovery statistics
- Priority-based resource recreation

**`hot_reload.zig`**
- `ShaderWatcher`: File change detection
- `HotReloadShader`: Auto-reloading shader
- Development workflow support

**`resource_aliasing.zig`**
- `AliasingPool`: Memory aliasing for temporaries
- `ResourceLifetime`: Lifetime tracking
- `AliasedResource`: Overlapping allocations

**`debug_labels.zig`**
- `DebugUtils`: RenderDoc/NSight integration
- `DebugCategory`, `DebugColors`: Label organization
- `ScopedLabel`: RAII-style labels
- `LabeledRecorder`: Labeled command recording

### `src/descriptor/` - Descriptor Management

**`template.zig`**
- `UpdateTemplate`: Descriptor update templates
- `TemplateBuilder`: Template construction
- `TemplateCache`: Template reuse
- `PushDescriptorUpdater`: Push descriptor support

### `src/memory/` - Memory Pools

**`usage_pools.zig`**
- `RingBufferPool`: Per-frame allocations
- `BlockPool`: Fixed-size block allocations
- `LinearPool`: Bump allocator
- `PoolManager`: Multi-pool management
- `UsagePattern`: Allocation pattern hints

### `src/command/` - Command Recording

**`batch.zig`**
- `BatchRecorder`: Batch command recording
- `ReusableCommandBuffer`: Command buffer reuse
- `ResourceAccess`: Access tracking
- `DispatchConfig`: Dispatch configuration

### `src/pipeline/` - Pipeline Management

**`variants.zig`**
- `ShaderVariantManager`: Specialization caching
- `VariantBuilder`: Variant construction
- `VariantSystem`: Variant system management
- `SpecConstantDef`: Constant definitions

## Design Principles

### 1. Explicit Resource Management

All resources follow the init/deinit pattern:

```zig
var buffer = try vulkan.Buffer.init(&ctx, size, .storage);
defer buffer.deinit(&ctx);
```

### 2. No Hidden State

- No global singletons (except error log for diagnostics)
- Multiple contexts can coexist
- All state is explicit and passed through function parameters

### 3. Graceful Degradation

The library never panics for recoverable conditions:

```zig
if (!vulkan.isAvailable()) {
    return error.VulkanUnavailable;
}
```

### 4. Dynamic Loading

Vulkan symbols are loaded dynamically:
- Build succeeds without Vulkan SDK
- Runtime detection of Vulkan availability
- Works with system-provided or container-provided Vulkan

### 5. Zero-Cost Abstractions

High-level types are thin wrappers:
- No hidden allocations
- Direct mapping to Vulkan concepts
- Inline function calls where possible

### 6. Production-Ready Error Handling

Comprehensive error context:
```zig
const ctx = vulkan.ErrorContext.from(
    vulkan.Error.OutOfDeviceMemory,
    "Buffer allocation failed",
    "Requested 1GB, available 512MB"
);
// ctx.location contains file:line:column
// vulkan.Diagnostics.getSuggestions(ctx.code) returns recovery hints
```

## Memory Management

### GPU Memory

The library provides multiple allocation strategies:

1. **SimpleAllocator**: One allocation per request
   - Simple but may waste memory
   - Good for prototyping and small workloads

2. **PooledAllocator**: Suballocation from large blocks
   - Efficient for production workloads
   - Tracks fragmentation
   - Supports multiple memory types

3. **RingBufferPool**: Frame-based allocation
   - Ideal for per-frame data
   - Automatic wrap-around

4. **LinearPool**: Bump allocator
   - Fast allocation, bulk reset
   - Good for temporary data

### Memory Usage Patterns

| Usage | Properties | Use Case |
|-------|-----------|----------|
| `gpu_only` | Device-local | Compute buffers |
| `cpu_to_gpu` | Host-visible, coherent | Staging/upload |
| `gpu_to_cpu` | Host-visible, cached | Readback |
| `cpu_and_gpu` | Host-visible, coherent | Convenience |

### Memory Pressure Handling

Automatic resource eviction when memory is constrained:

```zig
var handler = try vulkan.MemoryPressureHandler.init(allocator, &ctx, .{
    .warning_threshold = 0.8,
    .critical_threshold = 0.95,
    .auto_evict = true,
});

handler.registerResource(.{
    .name = "cache_buffer",
    .priority = 0,  // Lower = evicted first
    .evict_fn = my_evict_callback,
});
```

## Error Handling

### Error Categories

**Fatal (require context recreation):**
- `DeviceLost` - GPU crashed or hung
- `VulkanUnavailable` - No Vulkan runtime
- `IncompatibleDriver` - Driver incompatible

**Recoverable:**
- `OutOfHostMemory` - Reduce allocation size
- `OutOfDeviceMemory` - Free other resources
- `Timeout` - Retry operation

### Device Lost Recovery

Automatic recovery from GPU hangs:

```zig
var handler = try vulkan.DeviceLostHandler.init(allocator, .{
    .max_recovery_attempts = 3,
    .recreate_pipelines = true,
});

handler.registerResource("my_buffer", buffer_ptr, recreate_buffer_fn, priority);

// On device lost:
try handler.handleDeviceLost();  // Recreates resources by priority
```

## Synchronization Model

### CPU-GPU Sync
- Fences: Wait for GPU work to complete
- Memory barriers: Ensure memory visibility

### GPU-GPU Sync
- Semaphores: Order between queue submissions
- Timeline semaphores: Value-based synchronization
- Pipeline barriers: Order within command buffer

### Multi-Queue Sync

Cross-queue ownership transfers:

```zig
var sync = vulkan.QueueSync.init();
sync.addBuffer(buffer, .compute, .transfer);
sync.recordReleaseBarriers(compute_cmd, src_access, dst_access, src_stage, dst_stage);
// ... submit to transfer queue ...
sync.recordAcquireBarriers(transfer_cmd, src_access, dst_access, src_stage, dst_stage);
```

### Async Compute

Overlapping compute and transfer:

```zig
var coordinator = try vulkan.AsyncCoordinator.init(&ctx, allocator, .{
    .frame_count = 2,  // Double buffering
});

coordinator.beginFrame();
coordinator.scheduleTransfer(staging, gpu_buffer, .upload);
coordinator.scheduleCompute(64, 1, 1);
coordinator.scheduleTransfer(gpu_buffer, readback, .download);
coordinator.endFrame();
```

## Compute Graph

DAG-based task scheduling with automatic barrier insertion:

```zig
var graph = vulkan.ComputeGraph.init(allocator, &ctx);

const buf_a = try graph.registerResource("input", input_buffer, size);
const buf_b = try graph.registerResource("output", output_buffer, size);

_ = try graph.addTask(.{
    .name = "process",
    .pipeline = pipeline,
    .workgroups = .{ 64, 1, 1 },
    .resources = &.{
        .{ .handle = buf_a, .access = .read },
        .{ .handle = buf_b, .access = .write },
    },
});

try graph.compile();  // Computes dependencies, barriers
try graph.execute(cmd);  // Records to command buffer
```

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux x86_64 | Full | Primary development platform |
| Linux aarch64 | Full | Tested on ARM servers |
| Windows x86_64 | Full | vulkan-1.dll |
| macOS arm64 | Partial | Requires MoltenVK |

## Shader Compilation

Runtime GLSL/HLSL to SPIR-V:

```zig
// GLSL
if (vulkan.glsl.isCompilerAvailable()) {
    const result = try vulkan.glsl.compile(allocator, .{
        .source = glsl_source,
        .stage = .compute,
    });
    defer result.deinit();
}

// HLSL (via DXC)
if (vulkan.glsl.isHlslCompilerAvailable()) {
    const result = try vulkan.glsl.compileHlsl(allocator, .{
        .source = hlsl_source,
        .stage = .compute,
        .shader_model = .sm_6_5,
    });
}
```

## Extension Support

Query device capabilities:

```zig
const caps = vulkan.queryAllCapabilities(&ctx);

if (caps.subgroup.supported) {
    std.debug.print("Subgroup size: {}\n", .{caps.subgroup.subgroup_size});
}

if (caps.cooperative_matrix.supported) {
    // Use cooperative matrices for ML workloads
}
```

## CUDA Sibling Library

The API is designed to align with a planned CUDA library:

| Vulkan | CUDA |
|--------|------|
| Context | Context |
| Buffer | Buffer |
| ShaderModule | Module |
| ComputePipeline | Kernel |
| dispatch() | launch() |

This enables a future unified interface for GPU compute.
