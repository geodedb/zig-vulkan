//! Cooperative Matrix Example
//!
//! Demonstrates querying and using cooperative matrix (tensor core) capabilities
//! for accelerated matrix operations on supported hardware.
//!
//! Cooperative matrices enable efficient matrix multiplication operations
//! that leverage specialized tensor core hardware found in modern GPUs
//! (NVIDIA Tensor Cores, AMD Matrix Cores, Intel XMX).
//!
//! ## Hardware Requirements
//!
//! - NVIDIA: Volta (V100) or newer with VK_NV_cooperative_matrix
//! - AMD: RDNA3 or newer with VK_KHR_cooperative_matrix
//! - Intel: Arc GPUs with VK_KHR_cooperative_matrix
//!
//! ## Usage
//!
//! ```bash
//! zig build example-cooperative-matrix
//! ./zig-out/bin/example-cooperative-matrix
//! ```

const std = @import("std");
const vulkan = @import("vulkan");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Cooperative Matrix Example\n", .{});
    std.debug.print("==========================\n\n", .{});

    // Check Vulkan availability
    if (!vulkan.isAvailable()) {
        std.debug.print("Vulkan is not available on this system.\n", .{});
        std.debug.print("This example requires Vulkan to query cooperative matrix support.\n", .{});
        return;
    }

    // Probe for devices
    var probe = vulkan.probe(allocator);
    defer probe.deinit(allocator);

    if (!probe.isAvailable()) {
        std.debug.print("No Vulkan devices found.\n", .{});
        return;
    }

    std.debug.print("Found {} Vulkan device(s)\n\n", .{probe.devices.len});

    // Check each device for cooperative matrix support
    for (probe.devices, 0..) |device, i| {
        std.debug.print("Device {}: {s}\n", .{ i, device.getName() });
        std.debug.print("  Type: {s}\n", .{@tagName(device.device_type)});

        // Create context to query capabilities
        var ctx = vulkan.Context.init(allocator, .{
            .device_index = @intCast(i),
        }) catch |e| {
            std.debug.print("  Failed to create context: {}\n\n", .{e});
            continue;
        };
        defer ctx.deinit();

        // Query cooperative matrix capabilities
        const coop_caps = vulkan.queryCooperativeMatrixCapabilities(&ctx, allocator) catch |e| {
            std.debug.print("  Failed to query capabilities: {}\n\n", .{e});
            continue;
        };
        defer if (coop_caps.configurations.len > 0) allocator.free(coop_caps.configurations);

        if (!coop_caps.isAvailable()) {
            std.debug.print("  Cooperative Matrices: NOT SUPPORTED\n", .{});
            std.debug.print("  - Hardware may lack tensor cores\n", .{});
            std.debug.print("  - Requires VK_KHR_cooperative_matrix extension\n\n", .{});
            continue;
        }

        std.debug.print("  Cooperative Matrices: SUPPORTED\n", .{});
        std.debug.print("  Extension: VK_KHR_cooperative_matrix\n", .{});
        std.debug.print("  Robust Buffer Access: {s}\n", .{if (coop_caps.robust_buffer_access) "YES" else "NO"});

        // Print supported matrix configurations
        std.debug.print("\n  Supported Matrix Configurations:\n", .{});
        std.debug.print("  --------------------------------\n", .{});

        if (coop_caps.configurations.len > 0) {
            for (coop_caps.configurations) |config| {
                std.debug.print("    M={d:>2} N={d:>2} K={d:>2}  A:{s:<8} B:{s:<8} C:{s:<8} Result:{s:<8} Scope:{s}\n", .{
                    config.m,
                    config.n,
                    config.k,
                    formatComponentTypeKHR(config.a_type),
                    formatComponentTypeKHR(config.b_type),
                    formatComponentTypeKHR(config.c_type),
                    formatComponentTypeKHR(config.result_type),
                    @tagName(config.scope),
                });
            }
        } else {
            std.debug.print("    No specific configurations reported\n", .{});
            std.debug.print("    (Device may support cooperative matrices but not report configs)\n", .{});
        }

        // Example usage guidance
        std.debug.print("\n  Usage Notes:\n", .{});
        std.debug.print("  - Use 16x16x16 or 8x8x8 tile sizes for best performance\n", .{});
        std.debug.print("  - Ensure workgroup size matches cooperative matrix requirements\n", .{});
        std.debug.print("  - FP16 accumulation typically offers 2x throughput vs FP32\n", .{});

        std.debug.print("\n", .{});
    }

    // Show example GLSL code
    printExampleGlsl();
}

fn formatComponentTypeKHR(comp_type: vulkan.vk.types.ComponentTypeKHR) []const u8 {
    return switch (comp_type) {
        .float16 => "f16",
        .float32 => "f32",
        .float64 => "f64",
        .sint8 => "i8",
        .sint16 => "i16",
        .sint32 => "i32",
        .sint64 => "i64",
        .uint8 => "u8",
        .uint16 => "u16",
        .uint32 => "u32",
        .uint64 => "u64",
        _ => "???",
    };
}

fn printExampleGlsl() void {
    std.debug.print(
        \\Example GLSL (VK_KHR_cooperative_matrix):
        \\=========================================
        \\
        \\#version 450
        \\#extension GL_KHR_cooperative_matrix : require
        \\#extension GL_EXT_shader_explicit_arithmetic_types : require
        \\
        \\layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
        \\
        \\// Declare cooperative matrix types
        \\// coopmat<element_type, scope, rows, cols, use>
        \\
        \\void main() {{
        \\    // Matrix A: 16x16, float16, for matrix A operand
        \\    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matA;
        \\
        \\    // Matrix B: 16x16, float16, for matrix B operand
        \\    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matB;
        \\
        \\    // Accumulator: 16x16, float32
        \\    coopmat<float32_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> matC;
        \\
        \\    // Load from global memory
        \\    coopMatLoad(matA, inputA, element, stride, gl_CooperativeMatrixLayoutRowMajor);
        \\    coopMatLoad(matB, inputB, element, stride, gl_CooperativeMatrixLayoutRowMajor);
        \\    coopMatLoad(matC, inputC, element, stride, gl_CooperativeMatrixLayoutRowMajor);
        \\
        \\    // Matrix multiply-add: C = A * B + C
        \\    matC = coopMatMulAdd(matA, matB, matC);
        \\
        \\    // Store result
        \\    coopMatStore(matC, output, element, stride, gl_CooperativeMatrixLayoutRowMajor);
        \\}}
        \\
        \\Performance Tips:
        \\- Use subgroup size that matches hardware (typically 32 for NVIDIA, 64 for AMD)
        \\- Prefer FP16 inputs with FP32 accumulation for best throughput
        \\- Tile your matrices to fit cooperative matrix dimensions
        \\- Ensure memory alignment for coalesced loads/stores
        \\
    , .{});
}
