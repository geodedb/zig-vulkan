//! ML Inference Example
//!
//! Demonstrates GPU-accelerated machine learning inference using Vulkan compute.
//! Shows buffer layout and dispatch patterns for neural network operations:
//! - Matrix multiplication (dense layers)
//! - Activation functions (ReLU, sigmoid, softmax)
//! - Batch normalization
//! - Convolution operations

const std = @import("std");
const vulkan = @import("vulkan");

/// Neural network layer configuration
const LayerConfig = struct {
    input_size: u32,
    output_size: u32,
    batch_size: u32,

    fn weightsSize(self: LayerConfig) usize {
        return self.input_size * self.output_size;
    }

    fn biasSize(self: LayerConfig) usize {
        return self.output_size;
    }

    fn activationsSize(self: LayerConfig) usize {
        return self.batch_size * self.output_size;
    }
};

/// Network architecture for demonstration
const NetworkConfig = struct {
    layers: []const LayerConfig,

    fn totalParameters(self: NetworkConfig) usize {
        var total: usize = 0;
        for (self.layers) |layer| {
            total += layer.weightsSize() + layer.biasSize();
        }
        return total;
    }

    fn totalActivations(self: NetworkConfig) usize {
        var total: usize = 0;
        for (self.layers) |layer| {
            total += layer.activationsSize();
        }
        return total;
    }
};

/// Push constants for dense layer
const DenseLayerPushConstants = extern struct {
    input_size: u32,
    output_size: u32,
    batch_size: u32,
    activation: u32, // 0=none, 1=relu, 2=sigmoid, 3=tanh
};

/// Activation function types
const Activation = enum(u32) {
    none = 0,
    relu = 1,
    sigmoid = 2,
    tanh = 3,
    softmax = 4,

    fn getName(self: Activation) []const u8 {
        return switch (self) {
            .none => "Linear",
            .relu => "ReLU",
            .sigmoid => "Sigmoid",
            .tanh => "Tanh",
            .softmax => "Softmax",
        };
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("ML Inference Example\n", .{});
    std.debug.print("====================\n\n", .{});

    // Check availability
    if (!vulkan.isAvailable()) {
        std.debug.print("Vulkan not available\n", .{});
        return;
    }

    // Create context
    var ctx = vulkan.Context.init(allocator, .{
        .app_name = "ml_inference_example",
        .enable_validation = @import("builtin").mode == .Debug,
    }) catch |e| {
        std.debug.print("Failed to create context: {s}\n", .{vulkan.vk.describe(e)});
        return;
    };
    defer ctx.deinit();

    const device_info = ctx.deviceInfo();
    std.debug.print("Device: {s}\n", .{device_info.getName()});
    std.debug.print("Memory: {} MB\n\n", .{device_info.total_memory / (1024 * 1024)});

    // Example network architecture (simplified ResNet-like)
    const BATCH_SIZE: u32 = 32;
    const layers = [_]LayerConfig{
        .{ .input_size = 224 * 224 * 3, .output_size = 1024, .batch_size = BATCH_SIZE }, // Input -> FC1
        .{ .input_size = 1024, .output_size = 512, .batch_size = BATCH_SIZE }, // FC1 -> FC2
        .{ .input_size = 512, .output_size = 256, .batch_size = BATCH_SIZE }, // FC2 -> FC3
        .{ .input_size = 256, .output_size = 1000, .batch_size = BATCH_SIZE }, // FC3 -> Output (1000 classes)
    };

    const network = NetworkConfig{ .layers = &layers };

    std.debug.print("Network Architecture:\n", .{});
    std.debug.print("-" ** 60 ++ "\n", .{});
    std.debug.print("Batch size: {}\n\n", .{BATCH_SIZE});

    var total_flops: u64 = 0;
    for (layers, 0..) |layer, i| {
        const weights = layer.weightsSize();
        const bias = layer.biasSize();
        const flops = @as(u64, layer.batch_size) * layer.input_size * layer.output_size * 2;
        total_flops += flops;

        std.debug.print("Layer {}: {}x{} -> {}x{}\n", .{
            i + 1,
            layer.batch_size,
            layer.input_size,
            layer.batch_size,
            layer.output_size,
        });
        std.debug.print("  Weights: {} ({} MB)\n", .{ weights, weights * @sizeOf(f32) / (1024 * 1024) });
        std.debug.print("  Bias: {}\n", .{bias});
        std.debug.print("  FLOPs: {} ({} MFLOP)\n\n", .{ flops, flops / 1_000_000 });
    }

    const total_params = network.totalParameters();
    const total_params_bytes = total_params * @sizeOf(f32);
    const total_activations = network.totalActivations();
    const total_activations_bytes = total_activations * @sizeOf(f32);

    std.debug.print("Network Summary:\n", .{});
    std.debug.print("-" ** 60 ++ "\n", .{});
    std.debug.print("Total parameters: {} ({} MB)\n", .{ total_params, total_params_bytes / (1024 * 1024) });
    std.debug.print("Total activations: {} ({} MB)\n", .{ total_activations, total_activations_bytes / (1024 * 1024) });
    std.debug.print("Total FLOPs per batch: {} ({} GFLOP)\n\n", .{ total_flops, total_flops / 1_000_000_000 });

    // Create buffers for weights, activations, and intermediate results
    std.debug.print("Creating GPU buffers...\n", .{});

    // Weights buffer (read-only after initial upload)
    var weights_buffer = try vulkan.Buffer.init(&ctx, total_params_bytes, .staging);
    defer weights_buffer.deinit(&ctx);

    // Input buffer
    const input_size = BATCH_SIZE * 224 * 224 * 3 * @sizeOf(f32);
    var input_buffer = try vulkan.Buffer.init(&ctx, input_size, .staging);
    defer input_buffer.deinit(&ctx);

    // Output buffer
    const output_size = BATCH_SIZE * 1000 * @sizeOf(f32);
    var output_buffer = try vulkan.Buffer.init(&ctx, output_size, .readback);
    defer output_buffer.deinit(&ctx);

    // Intermediate activations (ping-pong buffers)
    var activation_a = try vulkan.Buffer.init(&ctx, total_activations_bytes, .storage);
    defer activation_a.deinit(&ctx);

    var activation_b = try vulkan.Buffer.init(&ctx, total_activations_bytes, .storage);
    defer activation_b.deinit(&ctx);

    std.debug.print("  Weights:     {} MB (staging)\n", .{total_params_bytes / (1024 * 1024)});
    std.debug.print("  Input:       {} MB (staging)\n", .{input_size / (1024 * 1024)});
    std.debug.print("  Output:      {} MB (readback)\n", .{output_size / (1024 * 1024)});
    std.debug.print("  Activations: {} MB x2 (storage, ping-pong)\n\n", .{total_activations_bytes / (1024 * 1024)});

    // Performance estimation
    const memory_bandwidth_gb_s: f64 = 500.0; // Typical GPU memory bandwidth
    const compute_tflops: f64 = 10.0; // Typical GPU compute capability

    const memory_time_ms = @as(f64, @floatFromInt(total_params_bytes + total_activations_bytes * 2)) / (memory_bandwidth_gb_s * 1e9) * 1000.0;
    const compute_time_ms = @as(f64, @floatFromInt(total_flops)) / (compute_tflops * 1e12) * 1000.0;

    std.debug.print("Performance Estimation (assuming {} GB/s bandwidth, {} TFLOPS):\n", .{
        @as(u32, @intFromFloat(memory_bandwidth_gb_s)),
        @as(u32, @intFromFloat(compute_tflops)),
    });
    std.debug.print("-" ** 60 ++ "\n", .{});
    std.debug.print("  Memory bound time: {d:.3} ms\n", .{memory_time_ms});
    std.debug.print("  Compute bound time: {d:.3} ms\n", .{compute_time_ms});
    std.debug.print("  Estimated throughput: {d:.0} samples/sec\n\n", .{1000.0 / @max(memory_time_ms, compute_time_ms)});

    // Example shader for dense layer
    std.debug.print("Example Dense Layer GLSL Shader:\n", .{});
    std.debug.print("-" ** 60 ++ "\n", .{});
    std.debug.print(
        \\#version 450
        \\layout(local_size_x = 256) in;
        \\
        \\layout(set = 0, binding = 0) readonly buffer Weights {{ float w[]; }};
        \\layout(set = 0, binding = 1) readonly buffer Bias {{ float b[]; }};
        \\layout(set = 0, binding = 2) readonly buffer Input {{ float x[]; }};
        \\layout(set = 0, binding = 3) writeonly buffer Output {{ float y[]; }};
        \\
        \\layout(push_constant) uniform PC {{
        \\    uint in_size, out_size, batch_size, activation;
        \\}};
        \\
        \\void main() {{
        \\    uint idx = gl_GlobalInvocationID.x;
        \\    uint batch = idx / out_size;
        \\    uint out_idx = idx % out_size;
        \\
        \\    if (batch >= batch_size) return;
        \\
        \\    float sum = b[out_idx];
        \\    for (uint i = 0; i < in_size; i++) {{
        \\        sum += x[batch * in_size + i] * w[out_idx * in_size + i];
        \\    }}
        \\
        \\    // Apply activation
        \\    if (activation == 1) sum = max(0, sum);      // ReLU
        \\    else if (activation == 2) sum = 1/(1+exp(-sum)); // Sigmoid
        \\
        \\    y[idx] = sum;
        \\}}
        \\
    , .{});

    // Activation function examples
    std.debug.print("\nActivation Functions:\n", .{});
    const activations = [_]Activation{ .none, .relu, .sigmoid, .tanh, .softmax };
    for (activations) |act| {
        std.debug.print("  {} - {s}\n", .{ @intFromEnum(act), act.getName() });
    }

    std.debug.print("\nExample completed.\n", .{});
}
