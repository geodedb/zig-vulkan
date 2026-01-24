//! Vector Scale Example
//!
//! Demonstrates GPU compute with push constants: scaling a vector by a factor.
//! Shows how to use push constants for small, frequently-changing data.

const std = @import("std");
const vulkan = @import("vulkan");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Vector Scale Example\n", .{});
    std.debug.print("====================\n\n", .{});

    // Check availability
    if (!vulkan.isAvailable()) {
        std.debug.print("Vulkan not available\n", .{});
        return;
    }

    // Create context
    var ctx = vulkan.Context.init(allocator, .{
        .app_name = "vector_scale_example",
    }) catch |e| {
        std.debug.print("Failed to create context: {s}\n", .{vulkan.vk.describe(e)});
        return;
    };
    defer ctx.deinit();

    const info = ctx.deviceInfo();
    std.debug.print("Using device: {s}\n", .{info.getName()});
    std.debug.print("Max push constant size: {} bytes\n\n", .{128}); // Minimum guaranteed

    const vector_size: usize = 2048;
    const buffer_size = vector_size * @sizeOf(f32);
    const scale_factor: f32 = 2.5;

    // Create buffers
    std.debug.print("Creating buffers ({} elements)...\n", .{vector_size});

    var input = try vulkan.Buffer.init(&ctx, buffer_size, .staging);
    defer input.deinit(&ctx);

    var output = try vulkan.Buffer.init(&ctx, buffer_size, .readback);
    defer output.deinit(&ctx);

    // Initialize input data
    std.debug.print("Initializing data...\n", .{});

    var input_data: [vector_size]f32 = undefined;
    for (0..vector_size) |i| {
        input_data[i] = @floatFromInt(i);
    }

    try input.writeSlice(f32, &input_data);

    // Push constant structure
    const PushConstants = extern struct {
        scale: f32,
        offset: f32,
        count: u32,
        _padding: u32 = 0,
    };

    const push_data = PushConstants{
        .scale = scale_factor,
        .offset = 0.0,
        .count = @intCast(vector_size),
    };

    std.debug.print("\nPush constants:\n", .{});
    std.debug.print("  scale: {d:.2}\n", .{push_data.scale});
    std.debug.print("  offset: {d:.2}\n", .{push_data.offset});
    std.debug.print("  count: {}\n", .{push_data.count});
    std.debug.print("  size: {} bytes\n", .{@sizeOf(PushConstants)});

    // Note: In a real application, you would:
    // 1. Load a compiled SPIR-V shader with push constant support
    // 2. Create pipeline with push constant range
    // 3. Use pushConstants() before dispatch

    std.debug.print("\n", .{});
    std.debug.print("NOTE: This example demonstrates push constant usage.\n", .{});
    std.debug.print("Push constants are ideal for:\n", .{});
    std.debug.print("  - Small, frequently-changing data (< 128 bytes guaranteed)\n", .{});
    std.debug.print("  - Per-dispatch parameters like scale factors, offsets, counts\n", .{});
    std.debug.print("  - Data that changes every frame or dispatch\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Example GLSL shader:\n", .{});
    std.debug.print("  #version 450\n", .{});
    std.debug.print("  layout(local_size_x = 64) in;\n", .{});
    std.debug.print("  layout(push_constant) uniform PushConstants {{\n", .{});
    std.debug.print("      float scale;\n", .{});
    std.debug.print("      float offset;\n", .{});
    std.debug.print("      uint count;\n", .{});
    std.debug.print("  }} pc;\n", .{});
    std.debug.print("  layout(binding = 0) buffer Input {{ float data[]; }};\n", .{});
    std.debug.print("  layout(binding = 1) buffer Output {{ float result[]; }};\n", .{});
    std.debug.print("  void main() {{\n", .{});
    std.debug.print("      uint i = gl_GlobalInvocationID.x;\n", .{});
    std.debug.print("      if (i < pc.count) {{\n", .{});
    std.debug.print("          result[i] = data[i] * pc.scale + pc.offset;\n", .{});
    std.debug.print("      }}\n", .{});
    std.debug.print("  }}\n", .{});

    // Show expected results
    std.debug.print("\nExpected output (first 8 elements, scale={d:.1}):\n", .{scale_factor});
    for (0..8) |i| {
        std.debug.print("  out[{}] = {} * {d:.1} = {d:.1}\n", .{
            i,
            input_data[i],
            scale_factor,
            input_data[i] * scale_factor,
        });
    }

    std.debug.print("\nExample completed.\n", .{});
}
