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

    // Thread-pool parallelism option
    const enable_tp = b.option(bool, "thread_pool", "Enable thread-pool-based parallelism") orelse true;

    // -Dwith_otel: compile in otel instrumentation. OFF by default;
    // when off the otel dependency is never resolved and the seam
    // collapses to a comptime-known no-op (verified by
    // otel/scripts/verify-consumer-noop.sh).
    const with_otel = b.option(
        bool,
        "with_otel",
        "Compile in otel instrumentation (off by default; zero overhead when off)",
    ) orelse false;

    // Forward otel build options so the upstream package is configured
    // identically across all consumers in the build graph. Without this,
    // each consumer's `b.lazyDependency("otel", ...)` materialises a
    // DISTINCT otel Module instance (different `build_options` ⇒ different
    // package keys), and zig 0.16 errors with `file exists in modules
    // 'otel' and 'otel0'` when transitive imports overlap.
    const otel_use_devnw_tls = b.option(
        bool,
        "otel_use_devnw_tls",
        "Forward to upstream otel: HTTPS via DevNW tls",
    ) orelse false;
    const otel_disabled = b.option(
        bool,
        "otel_disabled",
        "Forward to upstream otel: comptime-disable",
    ) orelse false;
    const otel_profiles = b.option(
        bool,
        "otel_profiles",
        "Forward to upstream otel: experimental profiles",
    ) orelse false;
    const otel_use_devnw_grpc = b.option(
        bool,
        "otel_use_devnw_grpc",
        "Forward to upstream otel: DevNW gRPC",
    ) orelse false;

    // Get thread_pool dependency if enabled
    const tp_mod: ?*std.Build.Module = if (enable_tp) blk: {
        const tp_dep = b.dependency("thread_pool", .{
            .target = target,
            .optimize = optimize,
        });
        break :blk tp_dep.module("thread_pool");
    } else null;

    // Parallel module
    const parallel_mod: ?*std.Build.Module = if (enable_tp) blk: {
        break :blk b.addModule("parallel", .{
            .root_source_file = b.path("src/parallel.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{},
        });
    } else null;
    if (parallel_mod) |pm| {
        if (tp_mod) |tm| pm.addImport("thread_pool", tm);
    }

    // ==========================================================================
    // Observability seam (otel)
    // ==========================================================================
    //
    // Materialize ONE shared `build_options` module. Each consumer module
    // imports the SAME module instance via `addImport` to avoid Zig 0.16's
    // "file exists in modules X and X0" error when two independent
    // `addOptions` modules end up sharing a transitive file path.
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "package_name", "vulkan");
    build_options.addOption(bool, "with_otel", with_otel);
    const build_options_mod = build_options.createModule();

    // Observability seam (otel). Declared as a SHARED NAMED MODULE so
    // every consumer (lib, tests, examples, benchmarks, plus the seam
    // tests) resolves to the same source file via
    // `@import("observability")`. Using a relative
    // `@import("observability.zig")` from each module would trip Zig's
    // "file exists in modules X and Y" error because the file would be
    // pulled into multiple module trees. The seam's `enabled` flag
    // tracks `build_options.with_otel`; when off, every helper compiles
    // to a comptime-known no-op and the otel dependency is never
    // resolved.
    const observability_mod = b.addModule("observability", .{
        .root_source_file = b.path("src/observability.zig"),
        .target = target,
        .optimize = optimize,
    });
    observability_mod.addImport("build_options", build_options_mod);

    // Gated otel dependency. The `lazy = true` entry in build.zig.zon
    // means Zig only fetches the package when this branch executes;
    // with the flag off the dependency is never resolved and the
    // produced artifacts contain zero otel symbols (verified by
    // otel/scripts/verify-consumer-noop.sh).
    var otel_mod: ?*std.Build.Module = null;
    var observability_init_mod: ?*std.Build.Module = null;
    if (with_otel) {
        if (b.lazyDependency("otel", .{
            .target = target,
            .optimize = optimize,
            .otel_use_devnw_tls = otel_use_devnw_tls,
            .otel_disabled = otel_disabled,
            .otel_profiles = otel_profiles,
            .otel_use_devnw_grpc = otel_use_devnw_grpc,
        })) |otel_dep| {
            const m = otel_dep.module("otel");
            otel_mod = m;
            // The seam itself imports `otel` when enabled.
            observability_mod.addImport("otel", m);

            // Bootstrap helper. Only registered when `-Dwith_otel=true`
            // so the disabled build never compiles otel-touching code.
            // Gated from lib.zig via
            // `if (observability.enabled) @import("observability_init")`.
            const init_mod = b.addModule("observability_init", .{
                .root_source_file = b.path("src/observability_init.zig"),
                .target = target,
                .optimize = optimize,
            });
            init_mod.addImport("otel", m);
            init_mod.addImport("observability", observability_mod);
            observability_init_mod = init_mod;
        }
    }

    // ==========================================================================
    // Library Module
    // ==========================================================================

    const vulkan_mod = b.addModule("vulkan", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{},
    });
    if (parallel_mod) |pm| vulkan_mod.addImport("parallel", pm);
    vulkan_mod.addImport("observability", observability_mod);
    vulkan_mod.addImport("build_options", build_options_mod);
    if (otel_mod) |m| vulkan_mod.addImport("otel", m);
    if (observability_init_mod) |m| vulkan_mod.addImport("observability_init", m);

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
    if (parallel_mod) |pm| lib.root_module.addImport("parallel", pm);
    lib.root_module.addImport("observability", observability_mod);
    lib.root_module.addImport("build_options", build_options_mod);
    if (otel_mod) |m| lib.root_module.addImport("otel", m);
    if (observability_init_mod) |m| lib.root_module.addImport("observability_init", m);

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
    if (parallel_mod) |pm| lib_tests.root_module.addImport("parallel", pm);
    lib_tests.root_module.addImport("observability", observability_mod);
    lib_tests.root_module.addImport("build_options", build_options_mod);
    if (otel_mod) |m| lib_tests.root_module.addImport("otel", m);
    if (observability_init_mod) |m| lib_tests.root_module.addImport("observability_init", m);
    const run_lib_tests = b.addRunArtifact(lib_tests);
    test_step.dependOn(&run_lib_tests.step);

    // Observability seam tests. Two test roots:
    //
    //   - observability_disabled_test.zig: runs under BOTH flag values.
    //     Each test inside it is skip-guarded via
    //     `if (vulkan.observability.enabled) return error.SkipZigTest;`
    //     so the disabled-mode invariants only fire when the flag is off.
    //
    //   - observability_enabled_test.zig: registered under both flag
    //     values too, but every test inside it skips when the flag is
    //     off. The `otel` import is only wired when `with_otel = true`;
    //     the test file's `comptime if (build_options.with_otel)` stub
    //     keeps it compiling under the disabled build.
    const disabled_obs_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/observability_disabled_test.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
                .{ .name = "build_options", .module = build_options_mod },
            },
        }),
    });
    const run_disabled_obs_tests = b.addRunArtifact(disabled_obs_tests);
    test_step.dependOn(&run_disabled_obs_tests.step);

    const enabled_obs_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/observability_enabled_test.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "vulkan", .module = vulkan_mod },
                .{ .name = "build_options", .module = build_options_mod },
            },
        }),
    });
    if (otel_mod) |m| enabled_obs_tests.root_module.addImport("otel", m);
    const run_enabled_obs_tests = b.addRunArtifact(enabled_obs_tests);
    test_step.dependOn(&run_enabled_obs_tests.step);

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
    if (parallel_mod) |pm| docs.root_module.addImport("parallel", pm);
    docs.root_module.addImport("observability", observability_mod);
    docs.root_module.addImport("build_options", build_options_mod);
    if (otel_mod) |m| docs.root_module.addImport("otel", m);
    if (observability_init_mod) |m| docs.root_module.addImport("observability_init", m);

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
        const output_name = std.mem.cutSuffix(u8, shader_src, ".glsl") orelse shader_src;
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
    const rm_out = b.addSystemCommand(&.{ "rm", "-rf", "zig-out" });
    clean_step.dependOn(&rm_out.step);
    const rm_cache = b.addSystemCommand(&.{ "rm", "-rf", ".zig-cache" });
    clean_step.dependOn(&rm_cache.step);
}
