// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Vulkan Compute Library Build Script
//!
//! This build script compiles the Vulkan compute library and related
//! tests, benchmarks, and examples.

const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard build options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build options
    const enable_shaderc = b.option(bool, "shaderc", "Enable shaderc for GLSL compilation") orelse false;
    _ = enable_shaderc;

    // Spice parallelism option
    const enable_spice = b.option(bool, "spice", "Enable spice-based parallelism") orelse true;

    // Build options module
    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_spice", enable_spice);

    // Get spice dependency if enabled
    const spice_mod: ?*std.Build.Module = if (enable_spice) blk: {
        const spice_dep = b.dependency("spice", .{
            .target = target,
            .optimize = optimize,
        });
        break :blk spice_dep.module("spice");
    } else null;

    // Parallel module
    const parallel_mod = b.addModule("parallel", .{
        .root_source_file = b.path("src/parallel.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_options.createModule() },
        },
    });
    if (spice_mod) |sm| {
        parallel_mod.addImport("spice", sm);
    }

    // ==========================================================================
    // Library Module
    // ==========================================================================

    const vulkan_mod = b.addModule("vulkan", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "parallel", .module = parallel_mod },
        },
    });

    // ==========================================================================
    // Static Library (optional)
    // ==========================================================================

    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "vulkan",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });

    b.installArtifact(lib);

    // ==========================================================================
    // Unit Tests
    // ==========================================================================

    const test_step = b.step("test", "Run unit tests");

    // Main library tests
    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);
    test_step.dependOn(&run_lib_tests.step);

    // ==========================================================================
    // Integration Tests
    // ==========================================================================

    const integration_test_step = b.step("test-integration", "Run integration tests (requires Vulkan)");

    const integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/integration.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
            },
        }),
    });
    const run_integration_tests = b.addRunArtifact(integration_tests);
    integration_test_step.dependOn(&run_integration_tests.step);

    // ==========================================================================
    // Benchmarks
    // ==========================================================================

    const bench_step = b.step("bench", "Run benchmarks");

    const bench_exe = b.addExecutable(.{
        .name = "bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/main.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
            },
        }),
    });
    bench_exe.linkLibC();

    const run_bench = b.addRunArtifact(bench_exe);
    if (b.args) |args| {
        run_bench.addArgs(args);
    }
    bench_step.dependOn(&run_bench.step);

    b.installArtifact(bench_exe);

    // ==========================================================================
    // Examples
    // ==========================================================================

    const examples_step = b.step("examples", "Build all examples");

    // Vector Add example
    const vector_add_exe = b.addExecutable(.{
        .name = "vector_add",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/vector_add.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
            },
        }),
    });
    vector_add_exe.linkLibC();
    b.installArtifact(vector_add_exe);
    examples_step.dependOn(&vector_add_exe.step);

    // Vector Scale example
    const vector_scale_exe = b.addExecutable(.{
        .name = "vector_scale",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/vector_scale.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
            },
        }),
    });
    vector_scale_exe.linkLibC();
    b.installArtifact(vector_scale_exe);
    examples_step.dependOn(&vector_scale_exe.step);

    // Probe example (always buildable, no shader needed)
    const probe_exe = b.addExecutable(.{
        .name = "probe",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/probe.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
            },
        }),
    });
    probe_exe.linkLibC();
    b.installArtifact(probe_exe);
    examples_step.dependOn(&probe_exe.step);

    // Run probe example
    const run_probe = b.addRunArtifact(probe_exe);
    const probe_step = b.step("probe", "Run the probe example to check Vulkan availability");
    probe_step.dependOn(&run_probe.step);

    // Matrix Multiply example
    const matrix_multiply_exe = b.addExecutable(.{
        .name = "matrix_multiply",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/matrix_multiply.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
            },
        }),
    });
    matrix_multiply_exe.linkLibC();
    b.installArtifact(matrix_multiply_exe);
    examples_step.dependOn(&matrix_multiply_exe.step);

    // Image Processing example
    const image_processing_exe = b.addExecutable(.{
        .name = "image_processing",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/image_processing.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
            },
        }),
    });
    image_processing_exe.linkLibC();
    b.installArtifact(image_processing_exe);
    examples_step.dependOn(&image_processing_exe.step);

    // ML Inference example
    const ml_inference_exe = b.addExecutable(.{
        .name = "ml_inference",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/ml_inference.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
            },
        }),
    });
    ml_inference_exe.linkLibC();
    b.installArtifact(ml_inference_exe);
    examples_step.dependOn(&ml_inference_exe.step);

    // Extensions example
    const extensions_exe = b.addExecutable(.{
        .name = "extensions",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/extensions.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
            },
        }),
    });
    extensions_exe.linkLibC();
    b.installArtifact(extensions_exe);
    examples_step.dependOn(&extensions_exe.step);

    // Timeline Semaphore example
    const timeline_semaphore_exe = b.addExecutable(.{
        .name = "timeline_semaphore",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/timeline_semaphore.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
            },
        }),
    });
    timeline_semaphore_exe.linkLibC();
    b.installArtifact(timeline_semaphore_exe);
    examples_step.dependOn(&timeline_semaphore_exe.step);

    // Cooperative Matrix example
    const cooperative_matrix_exe = b.addExecutable(.{
        .name = "cooperative_matrix",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/cooperative_matrix.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
            },
        }),
    });
    cooperative_matrix_exe.linkLibC();
    b.installArtifact(cooperative_matrix_exe);
    examples_step.dependOn(&cooperative_matrix_exe.step);

    // Ray Tracing Capabilities example
    const ray_tracing_exe = b.addExecutable(.{
        .name = "ray_tracing",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/ray_tracing.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
            },
        }),
    });
    ray_tracing_exe.linkLibC();
    b.installArtifact(ray_tracing_exe);
    examples_step.dependOn(&ray_tracing_exe.step);

    // Mesh Shader Capabilities example
    const mesh_shader_exe = b.addExecutable(.{
        .name = "mesh_shader",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/mesh_shader.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
            },
        }),
    });
    mesh_shader_exe.linkLibC();
    b.installArtifact(mesh_shader_exe);
    examples_step.dependOn(&mesh_shader_exe.step);

    // Multi-Device example
    const multi_device_exe = b.addExecutable(.{
        .name = "multi_device",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/multi_device.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
            },
        }),
    });
    multi_device_exe.linkLibC();
    b.installArtifact(multi_device_exe);
    examples_step.dependOn(&multi_device_exe.step);

    // ==========================================================================
    // Format Check
    // ==========================================================================

    const fmt_step = b.step("fmt", "Format source files");
    const fmt = b.addFmt(.{
        .paths = &.{
            "src",
            "tests",
            "bench",
            "examples",
            "build.zig",
        },
    });
    fmt_step.dependOn(&fmt.step);

    const fmt_check_step = b.step("fmt-check", "Check source file formatting");
    const fmt_check = b.addFmt(.{
        .paths = &.{
            "src",
            "tests",
            "bench",
            "examples",
            "build.zig",
        },
        .check = true,
    });
    fmt_check_step.dependOn(&fmt_check.step);

    // ==========================================================================
    // All Tests
    // ==========================================================================

    const test_all_step = b.step("test-all", "Run all tests (unit + integration)");
    test_all_step.dependOn(test_step);
    test_all_step.dependOn(integration_test_step);

    // ==========================================================================
    // Documentation
    // ==========================================================================

    const docs_step = b.step("docs", "Generate documentation");

    const docs = b.addLibrary(.{
        .linkage = .static,
        .name = "vulkan-docs",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const install_docs = b.addInstallDirectory(.{
        .source_dir = docs.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });

    docs_step.dependOn(&install_docs.step);

    // ==========================================================================
    // Shader Compilation
    // ==========================================================================

    const shaders_step = b.step("shaders", "Compile GLSL shaders to SPIR-V");

    // Shader compilation helper using glslangValidator or glslc
    const shader_sources = [_][]const u8{
        "shaders/compute.glsl",
        "shaders/vector_add.glsl",
        "shaders/matrix_multiply.glsl",
    };

    for (shader_sources) |shader_src| {
        // Try glslangValidator first (Khronos reference compiler)
        const compile_cmd = b.addSystemCommand(&.{
            "glslangValidator",
            "-V", // Generate SPIR-V
            "--target-env",
            "vulkan1.1",
            "-S", "comp", // Compute shader stage
            "-o",
        });

        // Output path: replace .glsl with .spv
        const output_name = std.mem.trimRight(u8, shader_src, ".glsl");
        compile_cmd.addArg(b.fmt("{s}.spv", .{output_name}));
        compile_cmd.addFileArg(b.path(shader_src));

        shaders_step.dependOn(&compile_cmd.step);
    }

    // Alternative: Add a shader-check step that validates without external tools
    const shader_check_step = b.step("shader-check", "Validate shader source files exist");
    _ = shader_check_step; // Placeholder for future validation

    // ==========================================================================
    // Clean
    // ==========================================================================

    const clean_step = b.step("clean", "Remove build artifacts");
    clean_step.dependOn(&b.addRemoveDirTree(b.path("zig-out")).step);
    clean_step.dependOn(&b.addRemoveDirTree(b.path(".zig-cache")).step);
}
