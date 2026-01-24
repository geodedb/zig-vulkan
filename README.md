# Zig Vulkan Compute Library

A production-grade GPU compute library for Zig, providing a clean, safe API for Vulkan compute operations.

**Version: 0.6.0**

## Status

| Aspect | Info |
|--------|------|
| API Stability | Beta |
| Zig Version | 0.14.0+ |
| Platforms | Linux, Windows (Vulkan-capable) |

## Features

### Core Features

| Feature | Status |
|---------|--------|
| Dynamic Vulkan loading | :white_check_mark: |
| Cross-platform (Linux, Windows, macOS) | :white_check_mark: |
| Graceful degradation when Vulkan unavailable | :white_check_mark: |
| Compute pipeline creation and dispatch | :white_check_mark: |
| Buffer management (staging, storage, readback) | :white_check_mark: |
| SPIR-V shader module loading with validation | :white_check_mark: |
| Memory management with usage hints | :white_check_mark: |
| Command buffer recording | :white_check_mark: |
| Synchronization primitives | :white_check_mark: |
| Timestamp queries for benchmarking | :white_check_mark: |
| Validation layer support | :white_check_mark: |
| Specialization constants | :white_check_mark: |
| Push constants | :white_check_mark: |
| Indirect dispatch | :white_check_mark: |

### Advanced Features

| Feature | Status |
|---------|--------|
| Pooled memory allocator (suballocation) | :white_check_mark: |
| Multi-queue support | :white_check_mark: |
| Timeline semaphores | :white_check_mark: |
| Cross-queue synchronization | :white_check_mark: |
| Async compute coordinator | :white_check_mark: |
| Compute graph (DAG scheduling) | :white_check_mark: |
| Multi-device context | :white_check_mark: |
| Pipeline cache (with disk persistence) | :white_check_mark: |
| Pipeline statistics queries | :white_check_mark: |

### Production Features

| Feature | Status |
|---------|--------|
| Device lost recovery | :white_check_mark: |
| Memory pressure handling | :white_check_mark: |
| Memory budget monitoring | :white_check_mark: |
| Comprehensive error context | :white_check_mark: |
| Error diagnostics and suggestions | :white_check_mark: |

### Developer Experience

| Feature | Status |
|---------|--------|
| GLSL to SPIR-V compilation | :white_check_mark: |
| HLSL to SPIR-V compilation (DXC) | :white_check_mark: |
| Shader reflection | :white_check_mark: |
| Shader hot-reload | :white_check_mark: |
| Debug labels (RenderDoc/NSight) | :white_check_mark: |
| GPU profiling | :white_check_mark: |

### Extension Support

| Extension | Status |
|-----------|--------|
| Subgroup operations | :white_check_mark: |
| Descriptor indexing (bindless) | :white_check_mark: |
| Push descriptors | :white_check_mark: |
| Descriptor update templates | :white_check_mark: |
| Cooperative matrices | :white_check_mark: |
| Memory budget | :white_check_mark: |
| Ray tracing (query only) | :white_check_mark: |
| Mesh shaders (query only) | :white_check_mark: |

## Requirements

- Zig 0.15.1 or later
- Vulkan runtime (optional at compile time, required at runtime for GPU operations)

## Installation

Add to your `build.zig.zon`:

```zig
.dependencies = .{
    .vulkan = .{
        .url = "https://gitlab.com/devnw/zig/vulkan/-/archive/main/vulkan-main.tar.gz",
        .hash = "...",
    },
},
```

Then in your `build.zig`:

```zig
const vulkan_dep = b.dependency("vulkan", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("vulkan", vulkan_dep.module("vulkan"));
```

## Quick Start

```zig
const std = @import("std");
const vulkan = @import("vulkan");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Check if Vulkan is available
    if (!vulkan.isAvailable()) {
        std.debug.print("Vulkan not available\n", .{});
        return error.VulkanUnavailable;
    }

    // Create a compute context
    var ctx = try vulkan.Context.init(allocator, .{
        .enable_validation = true,  // Enable for debugging
        .prefer_discrete = true,    // Prefer dedicated GPU
    });
    defer ctx.deinit();

    // Print device info
    const info = ctx.deviceInfo();
    std.debug.print("Using: {s}\n", .{info.getName()});

    // Create buffers
    var staging = try vulkan.Buffer.init(&ctx, 4096, .staging);
    defer staging.deinit(&ctx);

    var storage = try vulkan.Buffer.init(&ctx, 4096, .storage);
    defer storage.deinit(&ctx);

    // Write data to staging buffer
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try staging.writeSlice(f32, &data);

    // Record and submit commands...
}
```

## Probing the System

Check Vulkan availability without creating a full context:

```zig
var result = vulkan.probe(allocator);
defer result.deinit(allocator);

if (result.isAvailable()) {
    std.debug.print("Found {} device(s)\n", .{result.device_count});
    for (result.devices) |dev| {
        std.debug.print("  {s} - {s}\n", .{
            dev.getName(),
            @tagName(dev.device_type),
        });
    }
}
```

## Project Structure

```
vulkan/
├── src/
│   ├── lib.zig              # Public API
│   ├── vk/                  # Low-level Vulkan bindings
│   │   ├── types.zig        # Type definitions
│   │   ├── loader.zig       # Dynamic loader
│   │   └── error.zig        # Error handling
│   ├── compute/             # High-level compute API
│   │   ├── context.zig      # Context management
│   │   ├── buffer.zig       # Buffer operations
│   │   ├── pipeline.zig     # Compute pipelines
│   │   ├── command.zig      # Command buffers
│   │   ├── queue.zig        # Multi-queue support
│   │   ├── extensions.zig   # Extension capabilities
│   │   ├── multi_device.zig # Multi-GPU support
│   │   ├── graph.zig        # Compute graph
│   │   ├── async_compute.zig # Async coordination
│   │   └── profiling.zig    # GPU profiling
│   ├── util/                # Utilities
│   │   ├── allocator.zig    # Memory allocation
│   │   ├── pooled_allocator.zig  # Suballocation
│   │   ├── glsl.zig         # Shader compilation
│   │   ├── reflection.zig   # SPIR-V reflection
│   │   ├── error_recovery.zig # Device recovery
│   │   ├── hot_reload.zig   # Shader hot-reload
│   │   └── debug_labels.zig # Debug markers
│   ├── descriptor/          # Descriptor management
│   │   └── template.zig     # Update templates
│   ├── memory/              # Memory pools
│   │   └── usage_pools.zig  # Usage-based pools
│   ├── command/             # Command helpers
│   │   └── batch.zig        # Batch recording
│   └── pipeline/            # Pipeline helpers
│       └── variants.zig     # Shader variants
├── tests/                   # Integration tests
├── bench/                   # Benchmarks
├── examples/                # Example programs
└── docs/                    # Documentation
```

## Building

```bash
# Build library
zig build

# Run unit tests
zig build test

# Run integration tests (requires Vulkan)
zig build test-integration

# Build examples
zig build examples

# Run the probe example
zig build probe

# Run benchmarks
zig build bench
```

## Cross-Compilation

```bash
# Linux ARM64
zig build -Dtarget=aarch64-linux-gnu

# Windows
zig build -Dtarget=x86_64-windows-gnu

# macOS
zig build -Dtarget=aarch64-macos
```

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - Design and module structure

## API Overview

### Context

The main entry point for GPU compute operations.

```zig
var ctx = try vulkan.Context.init(allocator, .{
    .app_name = "my_app",
    .enable_validation = true,
    .prefer_discrete = true,
    .enable_timestamps = true,
});
defer ctx.deinit();
```

### Buffers

GPU memory buffers with different usage patterns:

| Usage | Description | Memory Type |
|-------|-------------|-------------|
| `.storage` | GPU read/write | Device-local |
| `.staging` | CPU to GPU upload | Host-visible |
| `.readback` | GPU to CPU download | Host-visible |
| `.uniform` | Constant data | Host-visible |

### Pipelines

Load SPIR-V shaders and create compute pipelines:

```zig
var shader = try vulkan.ShaderModule.initFromBytes(&ctx, @embedFile("shader.spv"));
defer shader.deinit();

var layout = try vulkan.DescriptorSetLayout.init(&ctx, &.{
    .{ .binding = 0, .descriptor_type = .storage_buffer },
    .{ .binding = 1, .descriptor_type = .storage_buffer },
});
defer layout.deinit();

var pipeline = try vulkan.ComputePipeline.init(&ctx, shader, "main", layout, null);
defer pipeline.deinit();
```

### Pipeline Cache

Disk-backed pipeline cache for faster startup:

```zig
var cache = try vulkan.PersistentPipelineCache.init(&ctx, .{
    .cache_path = "pipeline_cache.bin",
    .auto_save = true,
});
defer cache.deinit();  // Auto-saves on deinit

// Use cache for pipeline creation
var pipeline = try vulkan.ComputePipeline.initWithCache(&ctx, shader, "main", layout, null, cache.getHandle());
```

### Command Recording

Record and submit GPU commands:

```zig
var cmd = try vulkan.CommandBuffer.init(&ctx);
defer cmd.deinit();

try cmd.begin();
pipeline.bind(cmd.getHandle());
pipeline.bindDescriptorSet(cmd.getHandle(), descriptor_set);
pipeline.dispatch(cmd.getHandle(), 64, 1, 1);
try cmd.end();

try vulkan.Submission.submitAndWait(&ctx, cmd);
```

### Compute Graph

DAG-based task scheduling with automatic barriers:

```zig
var graph = vulkan.ComputeGraph.init(allocator, &ctx);
defer graph.deinit();

const input = try graph.registerResource("input", input_buffer, size);
const output = try graph.registerResource("output", output_buffer, size);

_ = try graph.addTask(.{
    .name = "process",
    .pipeline = pipeline,
    .workgroups = .{ 64, 1, 1 },
    .resources = &.{
        .{ .handle = input, .access = .read },
        .{ .handle = output, .access = .write },
    },
});

try graph.compile();
try graph.execute(cmd);
```

### Memory Pressure Handling

Automatic resource eviction when memory is constrained:

```zig
var handler = try vulkan.MemoryPressureHandler.init(allocator, &ctx, .{
    .warning_threshold = 0.8,
    .critical_threshold = 0.95,
    .auto_evict = true,
});
defer handler.deinit();

handler.registerResource(.{
    .name = "cache_buffer",
    .priority = 0,  // Lower priority = evicted first
    .evict_fn = my_evict_callback,
});
```

### Device Lost Recovery

Automatic recovery from GPU hangs:

```zig
var recovery = try vulkan.DeviceLostHandler.init(allocator, .{
    .max_recovery_attempts = 3,
    .recreate_pipelines = true,
});
defer recovery.deinit();

recovery.registerResource("my_buffer", buffer_ptr, recreate_buffer_fn, 10);

// On device lost, call:
try recovery.handleDeviceLost();  // Recreates resources by priority
```

### Shader Compilation

Runtime GLSL/HLSL to SPIR-V:

```zig
// GLSL
if (vulkan.glsl.isCompilerAvailable()) {
    const result = try vulkan.glsl.compile(allocator, .{
        .source = glsl_source,
        .stage = .compute,
    });
    defer result.deinit();
    // Use result.spirv
}

// HLSL (via DXC)
if (vulkan.glsl.isHlslCompilerAvailable()) {
    const result = try vulkan.glsl.compileHlsl(allocator, .{
        .source = hlsl_source,
        .stage = .compute,
        .shader_model = .sm_6_5,
    });
    defer result.deinit();
}
```

## Error Handling

All errors are returned as Zig errors with descriptive names:

```zig
const ctx = vulkan.Context.init(allocator, .{}) catch |err| {
    std.debug.print("Error: {s}\n", .{vulkan.describeError(err)});

    // Get recovery suggestions
    const suggestions = vulkan.Diagnostics.getSuggestions(err);
    for (suggestions) |s| {
        std.debug.print("  Suggestion: {s}\n", .{s});
    }
    return err;
};
```

Common errors:
- `VulkanUnavailable` - Vulkan loader not found
- `NoDevices` - No Vulkan devices found
- `NoComputeQueue` - No compute-capable queue
- `OutOfDeviceMemory` - GPU memory exhausted
- `DeviceLost` - GPU crashed or hung

## Extension Capabilities

Query device capabilities:

```zig
const caps = vulkan.queryAllCapabilities(&ctx);

if (caps.subgroup.supported) {
    std.debug.print("Subgroup size: {}\n", .{caps.subgroup.subgroup_size});
}

if (caps.cooperative_matrix.supported) {
    // Use cooperative matrices for ML workloads
}

// Memory budget
const budget = vulkan.queryMemoryBudget(&ctx);
for (0..budget.heap_count) |i| {
    std.debug.print("Heap {}: {} MB / {} MB\n", .{
        i,
        budget.heaps[i].usage / (1024 * 1024),
        budget.heaps[i].budget / (1024 * 1024),
    });
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Ensure tests pass: `zig build test`
4. Ensure formatting: `zig fmt --check src tests bench examples`
5. Submit a merge request

## License

Apache-2.0 License - see [LICENSE](LICENSE) for details.

## Related Projects

- [CUDA Compute Library](../cuda) - CUDA sibling library (planned)
- [Geode](https://gitlab.com/devnw/geode) - Graph database using this library
