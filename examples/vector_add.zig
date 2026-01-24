//! Vector Add Example
//!
//! Demonstrates basic GPU compute: adding two vectors element-wise.
//! This example requires a compute shader (vector_add.spv).

const std = @import("std");
const vulkan = @import("vulkan");

// Embedded SPIR-V shader (placeholder - would normally be compiled from GLSL)
// This is a minimal valid SPIR-V module that won't execute but allows
// the example to compile. For actual use, compile a real shader.
const vector_add_spirv = [_]u32{
    // SPIR-V magic number
    0x07230203, // Magic
    0x00010000, // Version 1.0
    0x00080001, // Generator
    0x00000001, // Bound
    0x00000000, // Schema
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Vector Add Example\n", .{});
    std.debug.print("==================\n\n", .{});

    // Check availability
    if (!vulkan.isAvailable()) {
        std.debug.print("Vulkan not available\n", .{});
        return;
    }

    // Create context
    var ctx = vulkan.Context.init(allocator, .{
        .app_name = "vector_add_example",
        .app_version = vulkan.vk.makeApiVersion(0, 1, 0, 0),
    }) catch |e| {
        std.debug.print("Failed to create context: {s}\n", .{vulkan.vk.describe(e)});
        return;
    };
    defer ctx.deinit();

    std.debug.print("Using device: {s}\n", .{ctx.deviceInfo().getName()});

    const vector_size: usize = 1024;
    const buffer_size = vector_size * @sizeOf(f32);

    // Create buffers
    std.debug.print("Creating buffers ({} elements)...\n", .{vector_size});

    var input_a = try vulkan.Buffer.init(&ctx, buffer_size, .staging);
    defer input_a.deinit(&ctx);

    var input_b = try vulkan.Buffer.init(&ctx, buffer_size, .staging);
    defer input_b.deinit(&ctx);

    var output = try vulkan.Buffer.init(&ctx, buffer_size, .readback);
    defer output.deinit(&ctx);

    // Initialize input data
    std.debug.print("Initializing data...\n", .{});

    var a_data: [vector_size]f32 = undefined;
    var b_data: [vector_size]f32 = undefined;

    for (0..vector_size) |i| {
        a_data[i] = @floatFromInt(i);
        b_data[i] = @floatFromInt(i * 2);
    }

    try input_a.writeSlice(f32, &a_data);
    try input_b.writeSlice(f32, &b_data);

    // Note: In a real application, you would:
    // 1. Load a compiled SPIR-V shader
    // 2. Create descriptor set layout
    // 3. Create compute pipeline
    // 4. Record and execute dispatch commands

    std.debug.print("\n", .{});
    std.debug.print("NOTE: This example demonstrates the API structure.\n", .{});
    std.debug.print("To run actual GPU compute, you need to:\n", .{});
    std.debug.print("  1. Compile a GLSL compute shader to SPIR-V\n", .{});
    std.debug.print("  2. Embed or load the SPIR-V binary\n", .{});
    std.debug.print("  3. Create pipeline and execute dispatch\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Example GLSL shader:\n", .{});
    std.debug.print("  #version 450\n", .{});
    std.debug.print("  layout(local_size_x = 64) in;\n", .{});
    std.debug.print("  layout(binding = 0) buffer A {{ float a[]; }};\n", .{});
    std.debug.print("  layout(binding = 1) buffer B {{ float b[]; }};\n", .{});
    std.debug.print("  layout(binding = 2) buffer C {{ float c[]; }};\n", .{});
    std.debug.print("  void main() {{\n", .{});
    std.debug.print("      uint i = gl_GlobalInvocationID.x;\n", .{});
    std.debug.print("      c[i] = a[i] + b[i];\n", .{});
    std.debug.print("  }}\n", .{});

    // Show expected results
    std.debug.print("\nExpected output (first 8 elements):\n", .{});
    for (0..8) |i| {
        std.debug.print("  c[{}] = {} + {} = {}\n", .{
            i,
            a_data[i],
            b_data[i],
            a_data[i] + b_data[i],
        });
    }

    std.debug.print("\nExample completed.\n", .{});
}
