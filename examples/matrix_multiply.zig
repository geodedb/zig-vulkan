//! Matrix Multiplication Example
//!
//! Demonstrates GPU matrix multiplication using Vulkan compute shaders.
//! Computes C = A * B for square matrices using a tiled algorithm.
//!
//! To use this example with actual GPU computation:
//! 1. Compile the matrix_multiply.comp shader to SPIR-V
//! 2. Embed or load the compiled SPIR-V
//! 3. Run the compute pipeline

const std = @import("std");
const vulkan = @import("vulkan");

/// Matrix size for demonstration (NxN)
const MATRIX_SIZE: usize = 1024;
const TILE_SIZE: usize = 16;

/// Push constants for the shader
const MatrixPushConstants = extern struct {
    m: u32, // Rows of A and C
    n: u32, // Columns of B and C
    k: u32, // Columns of A, Rows of B
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Matrix Multiplication Example\n", .{});
    std.debug.print("=============================\n\n", .{});
    std.debug.print("Matrix size: {}x{} (single precision float)\n", .{ MATRIX_SIZE, MATRIX_SIZE });
    std.debug.print("Total elements per matrix: {}\n", .{MATRIX_SIZE * MATRIX_SIZE});
    std.debug.print("Memory per matrix: {} MB\n\n", .{MATRIX_SIZE * MATRIX_SIZE * @sizeOf(f32) / (1024 * 1024)});

    // Check availability
    if (!vulkan.isAvailable()) {
        std.debug.print("Vulkan not available\n", .{});
        return;
    }

    // Create context
    var ctx = vulkan.Context.init(allocator, .{
        .app_name = "matrix_multiply_example",
        .enable_validation = @import("builtin").mode == .Debug,
    }) catch |e| {
        std.debug.print("Failed to create context: {s}\n", .{vulkan.vk.describe(e)});
        return;
    };
    defer ctx.deinit();

    const device_info = ctx.deviceInfo();
    std.debug.print("Using device: {s}\n", .{device_info.getName()});
    std.debug.print("Max workgroup size: {}x{}x{}\n\n", .{
        device_info.max_compute_work_group_size[0],
        device_info.max_compute_work_group_size[1],
        device_info.max_compute_work_group_size[2],
    });

    const matrix_elements = MATRIX_SIZE * MATRIX_SIZE;
    const matrix_bytes = matrix_elements * @sizeOf(f32);

    // Create buffers for A, B, and C matrices
    std.debug.print("Creating GPU buffers...\n", .{});

    var matrix_a = try vulkan.Buffer.init(&ctx, matrix_bytes, .staging);
    defer matrix_a.deinit(&ctx);

    var matrix_b = try vulkan.Buffer.init(&ctx, matrix_bytes, .staging);
    defer matrix_b.deinit(&ctx);

    var matrix_c = try vulkan.Buffer.init(&ctx, matrix_bytes, .readback);
    defer matrix_c.deinit(&ctx);

    std.debug.print("  Matrix A: {} bytes (staging)\n", .{matrix_bytes});
    std.debug.print("  Matrix B: {} bytes (staging)\n", .{matrix_bytes});
    std.debug.print("  Matrix C: {} bytes (readback)\n\n", .{matrix_bytes});

    // Initialize input matrices
    std.debug.print("Initializing matrices...\n", .{});

    var a_data: [MATRIX_SIZE * MATRIX_SIZE]f32 = undefined;
    var b_data: [MATRIX_SIZE * MATRIX_SIZE]f32 = undefined;

    // Initialize A as identity-like matrix (diagonal = 1, off-diagonal = 0.001)
    // Initialize B with incrementing values
    for (0..MATRIX_SIZE) |i| {
        for (0..MATRIX_SIZE) |j| {
            const idx = i * MATRIX_SIZE + j;
            a_data[idx] = if (i == j) 1.0 else 0.001;
            b_data[idx] = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(MATRIX_SIZE));
        }
    }

    try matrix_a.writeSlice(f32, &a_data);
    try matrix_b.writeSlice(f32, &b_data);

    // Calculate expected result for verification (first few elements)
    std.debug.print("\nExpected output (first 4x4 corner):\n", .{});
    for (0..4) |i| {
        std.debug.print("  ", .{});
        for (0..4) |j| {
            // C[i,j] = sum(A[i,k] * B[k,j])
            var sum: f32 = 0;
            for (0..MATRIX_SIZE) |k| {
                sum += a_data[i * MATRIX_SIZE + k] * b_data[k * MATRIX_SIZE + j];
            }
            std.debug.print("{d:8.4} ", .{sum});
        }
        std.debug.print("\n", .{});
    }

    // Calculate theoretical performance
    const flops_per_element: u64 = 2 * MATRIX_SIZE; // multiply + add per output element
    const total_flops: u64 = matrix_elements * flops_per_element;

    std.debug.print("\n", .{});
    std.debug.print("Theoretical compute:\n", .{});
    std.debug.print("  FLOPs: {} ({} GFLOP)\n", .{ total_flops, total_flops / 1_000_000_000 });
    std.debug.print("  Memory reads: {} bytes ({} MB)\n", .{ 2 * matrix_bytes, 2 * matrix_bytes / (1024 * 1024) });
    std.debug.print("  Memory writes: {} bytes ({} MB)\n", .{ matrix_bytes, matrix_bytes / (1024 * 1024) });

    // Dispatch dimensions
    const groups_x = (MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE;
    const groups_y = (MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE;

    std.debug.print("\nDispatch configuration:\n", .{});
    std.debug.print("  Workgroup size: {}x{}x1\n", .{ TILE_SIZE, TILE_SIZE });
    std.debug.print("  Dispatch groups: {}x{}x1\n", .{ groups_x, groups_y });
    std.debug.print("  Total threads: {}\n", .{groups_x * groups_y * TILE_SIZE * TILE_SIZE});

    // Push constants
    const push_constants = MatrixPushConstants{
        .m = @intCast(MATRIX_SIZE),
        .n = @intCast(MATRIX_SIZE),
        .k = @intCast(MATRIX_SIZE),
    };

    std.debug.print("\n", .{});
    std.debug.print("To complete this example with actual GPU execution:\n", .{});
    std.debug.print("  1. Compile shaders/matrix_multiply.comp to SPIR-V:\n", .{});
    std.debug.print("     glslc -O shaders/matrix_multiply.comp -o shaders/matrix_multiply.spv\n", .{});
    std.debug.print("  2. Load the SPIR-V and create a compute pipeline\n", .{});
    std.debug.print("  3. Bind descriptors and push constants\n", .{});
    std.debug.print("  4. Dispatch {} x {} x 1 workgroups\n", .{ groups_x, groups_y });
    std.debug.print("  5. Read back results from matrix_c\n", .{});
    std.debug.print("\nPush constants: M={}, N={}, K={}\n", .{
        push_constants.m,
        push_constants.n,
        push_constants.k,
    });

    std.debug.print("\nExample completed.\n", .{});
}
