//! Shader Compiler Detection Example
//!
//! Demonstrates runtime detection of GLSL and HLSL compilers.
//! This is useful for applications that want to compile shaders
//! at runtime rather than embedding pre-compiled SPIR-V.

const std = @import("std");
const vulkan = @import("vulkan");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Shader Compiler Detection Example\n", .{});
    std.debug.print("==================================\n\n", .{});

    // Detect available compilers
    const info = vulkan.glsl.detectCompilers(allocator);

    std.debug.print("GLSL Compiler (glslc/glslangValidator):\n", .{});
    std.debug.print("  Available: {s}\n", .{if (info.glsl_available) "YES" else "NO"});
    if (info.glsl_backend) |backend| {
        std.debug.print("  Backend: {s}\n", .{backend.executable()});
    }
    if (info.glsl_path) |path| {
        std.debug.print("  Path: {s}\n", .{path});
    }
    if (!info.glsl_available) {
        std.debug.print("  Install: Vulkan SDK (https://vulkan.lunarg.com/)\n", .{});
    }

    std.debug.print("\nHLSL Compiler (DXC):\n", .{});
    std.debug.print("  Available: {s}\n", .{if (info.hlsl_available) "YES" else "NO"});
    if (info.dxc_path) |path| {
        std.debug.print("  Path: {s}\n", .{path});
    }
    if (info.dxc_version) |ver| {
        std.debug.print("  Version: {s}\n", .{ver});
    }
    if (info.max_shader_model) |sm| {
        std.debug.print("  Max Shader Model: {s}\n", .{sm.toString()});
    }
    if (!info.hlsl_available) {
        std.debug.print("  Install: DirectXShaderCompiler (https://github.com/microsoft/DirectXShaderCompiler)\n", .{});
    }

    // Demonstrate unified compiler
    std.debug.print("\nUnified Compiler Support:\n", .{});
    var compiler = vulkan.glsl.UnifiedCompiler.init(allocator);
    defer compiler.deinit();

    std.debug.print("  GLSL support: {s}\n", .{if (compiler.supportsLanguage(.glsl)) "YES" else "NO"});
    std.debug.print("  HLSL support: {s}\n", .{if (compiler.supportsLanguage(.hlsl)) "YES" else "NO"});

    // Language detection from file extensions
    std.debug.print("\nLanguage Detection (by extension):\n", .{});
    const test_files = [_][]const u8{
        "shader.glsl",
        "compute.comp",
        "vertex.vert",
        "pixel.hlsl",
        "effect.fx",
        "unknown.xyz",
    };

    for (test_files) |path| {
        if (vulkan.glsl.detectLanguage(path)) |lang| {
            std.debug.print("  {s} -> {s}\n", .{ path, @tagName(lang) });
        } else {
            std.debug.print("  {s} -> unknown\n", .{path});
        }
    }

    // Example: Try to compile a simple GLSL shader if available
    if (info.glsl_available) {
        std.debug.print("\nTrying GLSL compilation...\n", .{});

        const glsl_source =
            \\#version 450
            \\layout(local_size_x = 64) in;
            \\layout(binding = 0) buffer Data { float data[]; };
            \\void main() {
            \\    uint idx = gl_GlobalInvocationID.x;
            \\    data[idx] = data[idx] * 2.0;
            \\}
        ;

        var result = vulkan.glsl.compile(allocator, .{
            .source = glsl_source,
            .stage = .compute,
            .entry_point = "main",
        }) catch |e| {
            std.debug.print("  Compilation failed: {}\n", .{e});
            return;
        };
        defer result.deinit();

        std.debug.print("  SUCCESS! Generated {} SPIR-V words\n", .{result.spirv.len});

        // Validate the output
        const validation = vulkan.validateSpirv(result.spirv);
        std.debug.print("  SPIR-V valid: {}\n", .{validation.is_valid});
    }

    std.debug.print("\nExample completed.\n", .{});
}
