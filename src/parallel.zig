// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Parallel Vulkan operations using spice heartbeat scheduling.
//!
//! Provides high-performance parallel implementations for CPU-side
//! Vulkan operations including shader variant precompilation,
//! multi-device context creation, and batch command preparation.
//!
//! ## Overview
//!
//! This module enables parallel processing of Vulkan CPU-side operations
//! using the spice thread pool. Ideal for game engines and graphics
//! applications that need to prepare many shader variants or command
//! buffers simultaneously.
//!
//! ## Build Configuration
//!
//! Enable: `-Dspice=true` (default)
//! Disable: `-Dspice=false`
//!
//! ## Usage Example
//!
//! ```zig
//! const parallel = @import("parallel.zig");
//!
//! // Precompile multiple shader variants in parallel
//! var variants: [8][]const parallel.SpecValue = undefined;
//! var results: [8]parallel.ShaderCompileResult = undefined;
//!
//! parallel.parallelShaderPrecompile(
//!     allocator,
//!     &variants,
//!     &results,
//!     myShaderCompiler,
//! );
//!
//! // Initialize multiple Vulkan devices in parallel
//! var device_configs: [4]parallel.DeviceConfig = undefined;
//! var device_results: [4]parallel.DeviceInitResult = undefined;
//!
//! parallel.parallelDeviceInit(
//!     allocator,
//!     &device_configs,
//!     &device_results,
//!     myDeviceInitializer,
//! );
//! ```
//!
//! ## Performance Thresholds
//!
//! | Operation        | Threshold   | Sequential Below |
//! |------------------|-------------|------------------|
//! | Shader Precompile| 2 variants  | Yes              |
//! | Device Init      | 2 devices   | Yes              |
//! | Command Prepare  | 4 batches   | Yes              |

const std = @import("std");
const build_options = @import("build_options");

const spice = if (build_options.enable_spice) @import("spice") else struct {
    pub const ThreadPool = void;
    pub const Task = void;
    pub fn Future(comptime I: type, comptime O: type) type {
        _ = I;
        _ = O;
        return void;
    }
};

// ============================================================================
// Global Thread Pool
// ============================================================================

var global_pool: if (build_options.enable_spice) ?spice.ThreadPool else void = if (build_options.enable_spice) null else {};
var pool_mutex: std.Thread.Mutex = .{};

pub fn getPool() if (build_options.enable_spice) *spice.ThreadPool else void {
    if (!build_options.enable_spice) return;

    pool_mutex.lock();
    defer pool_mutex.unlock();

    if (global_pool == null) {
        global_pool = spice.ThreadPool.init(std.heap.page_allocator);
        global_pool.?.start(.{});
    }
    return &global_pool.?;
}

pub fn deinitPool() void {
    if (!build_options.enable_spice) return;

    pool_mutex.lock();
    defer pool_mutex.unlock();

    if (global_pool) |*pool| {
        pool.deinit();
        global_pool = null;
    }
}

// ============================================================================
// Parallel Shader Variant Precompilation
// ============================================================================

/// Specialization constant value
pub const SpecValue = struct {
    constant_id: u32,
    value: u64,
};

/// Shader compilation result
pub const ShaderCompileResult = struct {
    success: bool,
    variant_index: usize,
    error_msg: ?[]const u8,
};

/// Shader compiler function type
pub const ShaderCompiler = *const fn (variant: []const SpecValue, allocator: std.mem.Allocator) ShaderCompileResult;

/// Context for parallel shader compilation
pub const ShaderCompileContext = struct {
    variants: []const []const SpecValue,
    results: []ShaderCompileResult,
    compiler: ShaderCompiler,
    allocator: std.mem.Allocator,
};

/// Precompile multiple shader variants in parallel.
///
/// Each shader variant compilation is independent, enabling
/// parallel generation of specialized pipelines.
///
/// Parameters:
/// - allocator: Memory allocator
/// - variants: Array of specialization constant sets
/// - results: Array to store compilation results
/// - compiler: Function to compile each variant
///
/// Threshold: Uses parallel processing for 2+ variants
pub fn parallelShaderPrecompile(
    allocator: std.mem.Allocator,
    variants: []const []const SpecValue,
    results: []ShaderCompileResult,
    compiler: ShaderCompiler,
) void {
    std.debug.assert(variants.len == results.len);

    if (!build_options.enable_spice or variants.len < 2) {
        for (variants, 0..) |variant, i| {
            results[i] = compiler(variant, allocator);
        }
        return;
    }

    const pool = getPool();
    const ctx = ShaderCompileContext{
        .variants = variants,
        .results = results,
        .compiler = compiler,
        .allocator = allocator,
    };
    _ = pool.call(void, parallelShaderCompileTask, ctx);
}

fn parallelShaderCompileTask(t: *spice.Task, ctx: ShaderCompileContext) void {
    const variants = ctx.variants;
    const results = ctx.results;

    if (variants.len < 2) {
        for (variants, 0..) |variant, i| {
            results[i] = ctx.compiler(variant, ctx.allocator);
        }
        return;
    }

    const mid = variants.len / 2;
    const left_ctx = ShaderCompileContext{
        .variants = variants[0..mid],
        .results = results[0..mid],
        .compiler = ctx.compiler,
        .allocator = ctx.allocator,
    };
    const right_ctx = ShaderCompileContext{
        .variants = variants[mid..],
        .results = results[mid..],
        .compiler = ctx.compiler,
        .allocator = ctx.allocator,
    };

    var fut = spice.Future(ShaderCompileContext, void).init();
    fut.fork(t, parallelShaderCompileTask, right_ctx);

    t.call(void, parallelShaderCompileTask, left_ctx);
    if (fut.join(t) == null) {
        t.call(void, parallelShaderCompileTask, right_ctx);
    }
}

// ============================================================================
// Parallel Multi-Device Context Creation
// ============================================================================

/// Device info for context creation
pub const DeviceInfo = struct {
    physical_device: u64,
    queue_family: u32,
    device_name: []const u8,
};

/// Context creation result
pub const ContextCreateResult = struct {
    success: bool,
    context_handle: ?u64,
    error_code: ?i32,
};

/// Context creator function type
pub const ContextCreator = *const fn (device: DeviceInfo, allocator: std.mem.Allocator) ContextCreateResult;

/// Context for parallel device context creation
pub const DeviceContextCreateContext = struct {
    devices: []const DeviceInfo,
    results: []ContextCreateResult,
    creator: ContextCreator,
    allocator: std.mem.Allocator,
};

/// Create contexts for multiple GPU devices in parallel.
///
/// Each device context creation is independent, enabling
/// parallel initialization across multiple GPUs.
///
/// Parameters:
/// - allocator: Memory allocator
/// - devices: Array of device info structures
/// - results: Array to store creation results
/// - creator: Function to create each context
///
/// Threshold: Uses parallel processing for 2+ devices
pub fn parallelContextCreate(
    allocator: std.mem.Allocator,
    devices: []const DeviceInfo,
    results: []ContextCreateResult,
    creator: ContextCreator,
) void {
    std.debug.assert(devices.len == results.len);

    if (!build_options.enable_spice or devices.len < 2) {
        for (devices, 0..) |device, i| {
            results[i] = creator(device, allocator);
        }
        return;
    }

    const pool = getPool();
    const ctx = DeviceContextCreateContext{
        .devices = devices,
        .results = results,
        .creator = creator,
        .allocator = allocator,
    };
    _ = pool.call(void, parallelContextCreateTask, ctx);
}

fn parallelContextCreateTask(t: *spice.Task, ctx: DeviceContextCreateContext) void {
    const devices = ctx.devices;
    const results = ctx.results;

    if (devices.len < 2) {
        for (devices, 0..) |device, i| {
            results[i] = ctx.creator(device, ctx.allocator);
        }
        return;
    }

    const mid = devices.len / 2;
    const left_ctx = DeviceContextCreateContext{
        .devices = devices[0..mid],
        .results = results[0..mid],
        .creator = ctx.creator,
        .allocator = ctx.allocator,
    };
    const right_ctx = DeviceContextCreateContext{
        .devices = devices[mid..],
        .results = results[mid..],
        .creator = ctx.creator,
        .allocator = ctx.allocator,
    };

    var fut = spice.Future(DeviceContextCreateContext, void).init();
    fut.fork(t, parallelContextCreateTask, right_ctx);

    t.call(void, parallelContextCreateTask, left_ctx);
    if (fut.join(t) == null) {
        t.call(void, parallelContextCreateTask, right_ctx);
    }
}

// ============================================================================
// Parallel Command Buffer Recording
// ============================================================================

/// Command buffer record input
pub const CommandBufferInput = struct {
    buffer_handle: u64,
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
    pipeline: u64,
};

/// Command buffer record result
pub const CommandBufferResult = struct {
    success: bool,
    error_code: ?i32,
};

/// Command recorder function type
pub const CommandRecorder = *const fn (input: CommandBufferInput) CommandBufferResult;

/// Context for parallel command recording
pub const CommandRecordContext = struct {
    inputs: []const CommandBufferInput,
    results: []CommandBufferResult,
    recorder: CommandRecorder,
};

/// Record multiple command buffers in parallel.
///
/// Each command buffer recording is independent, enabling
/// parallel preparation of compute dispatches.
///
/// Parameters:
/// - inputs: Array of command buffer inputs
/// - results: Array to store recording results
/// - recorder: Function to record each command buffer
///
/// Threshold: Uses parallel processing for 4+ buffers
pub fn parallelCommandRecord(
    inputs: []const CommandBufferInput,
    results: []CommandBufferResult,
    recorder: CommandRecorder,
) void {
    std.debug.assert(inputs.len == results.len);

    if (!build_options.enable_spice or inputs.len < 4) {
        for (inputs, 0..) |input, i| {
            results[i] = recorder(input);
        }
        return;
    }

    const pool = getPool();
    const ctx = CommandRecordContext{
        .inputs = inputs,
        .results = results,
        .recorder = recorder,
    };
    _ = pool.call(void, parallelCommandRecordTask, ctx);
}

fn parallelCommandRecordTask(t: *spice.Task, ctx: CommandRecordContext) void {
    const inputs = ctx.inputs;
    const results = ctx.results;

    if (inputs.len < 2) {
        for (inputs, 0..) |input, i| {
            results[i] = ctx.recorder(input);
        }
        return;
    }

    const mid = inputs.len / 2;
    const left_ctx = CommandRecordContext{
        .inputs = inputs[0..mid],
        .results = results[0..mid],
        .recorder = ctx.recorder,
    };
    const right_ctx = CommandRecordContext{
        .inputs = inputs[mid..],
        .results = results[mid..],
        .recorder = ctx.recorder,
    };

    var fut = spice.Future(CommandRecordContext, void).init();
    fut.fork(t, parallelCommandRecordTask, right_ctx);

    t.call(void, parallelCommandRecordTask, left_ctx);
    if (fut.join(t) == null) {
        t.call(void, parallelCommandRecordTask, right_ctx);
    }
}

// ============================================================================
// Parallel Buffer Memory Operations
// ============================================================================

/// Buffer copy operation
pub const BufferCopyOp = struct {
    src_buffer: u64,
    dst_buffer: u64,
    src_offset: u64,
    dst_offset: u64,
    size: u64,
};

/// Buffer copy result
pub const BufferCopyResult = struct {
    success: bool,
    bytes_copied: u64,
};

/// Buffer copier function type
pub const BufferCopier = *const fn (op: BufferCopyOp) BufferCopyResult;

/// Context for parallel buffer copies
pub const BufferCopyContext = struct {
    ops: []const BufferCopyOp,
    results: []BufferCopyResult,
    copier: BufferCopier,
};

/// Copy multiple buffers in parallel.
///
/// Each buffer copy is independent, enabling parallel
/// memory transfer preparation.
///
/// Parameters:
/// - ops: Array of buffer copy operations
/// - results: Array to store copy results
/// - copier: Function to perform each copy
///
/// Threshold: Uses parallel processing for 4+ copies
pub fn parallelBufferCopy(
    ops: []const BufferCopyOp,
    results: []BufferCopyResult,
    copier: BufferCopier,
) void {
    std.debug.assert(ops.len == results.len);

    if (!build_options.enable_spice or ops.len < 4) {
        for (ops, 0..) |op, i| {
            results[i] = copier(op);
        }
        return;
    }

    const pool = getPool();
    const ctx = BufferCopyContext{
        .ops = ops,
        .results = results,
        .copier = copier,
    };
    _ = pool.call(void, parallelBufferCopyTask, ctx);
}

fn parallelBufferCopyTask(t: *spice.Task, ctx: BufferCopyContext) void {
    const ops = ctx.ops;
    const results = ctx.results;

    if (ops.len < 2) {
        for (ops, 0..) |op, i| {
            results[i] = ctx.copier(op);
        }
        return;
    }

    const mid = ops.len / 2;
    const left_ctx = BufferCopyContext{
        .ops = ops[0..mid],
        .results = results[0..mid],
        .copier = ctx.copier,
    };
    const right_ctx = BufferCopyContext{
        .ops = ops[mid..],
        .results = results[mid..],
        .copier = ctx.copier,
    };

    var fut = spice.Future(BufferCopyContext, void).init();
    fut.fork(t, parallelBufferCopyTask, right_ctx);

    t.call(void, parallelBufferCopyTask, left_ctx);
    if (fut.join(t) == null) {
        t.call(void, parallelBufferCopyTask, right_ctx);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "parallel shader precompile" {
    if (!build_options.enable_spice) return error.SkipZigTest;

    const testCompiler = struct {
        fn compile(variant: []const SpecValue, allocator: std.mem.Allocator) ShaderCompileResult {
            _ = allocator;
            return .{
                .success = variant.len > 0,
                .variant_index = 0,
                .error_msg = null,
            };
        }
    }.compile;

    var variants: [4][]const SpecValue = undefined;
    var results: [4]ShaderCompileResult = undefined;

    const spec_val = SpecValue{ .constant_id = 0, .value = 1 };
    const spec_slice: []const SpecValue = &[_]SpecValue{spec_val};

    for (&variants) |*v| {
        v.* = spec_slice;
    }

    parallelShaderPrecompile(std.testing.allocator, &variants, &results, testCompiler);

    for (results) |result| {
        try std.testing.expect(result.success);
    }
}

test "parallel context create" {
    if (!build_options.enable_spice) return error.SkipZigTest;

    const testCreator = struct {
        fn create(device: DeviceInfo, allocator: std.mem.Allocator) ContextCreateResult {
            _ = allocator;
            return .{
                .success = device.physical_device != 0,
                .context_handle = device.physical_device,
                .error_code = null,
            };
        }
    }.create;

    var devices: [4]DeviceInfo = undefined;
    var results: [4]ContextCreateResult = undefined;

    for (&devices, 0..) |*d, i| {
        d.* = .{
            .physical_device = i + 1,
            .queue_family = 0,
            .device_name = "Test Device",
        };
    }

    parallelContextCreate(std.testing.allocator, &devices, &results, testCreator);

    for (results) |result| {
        try std.testing.expect(result.success);
    }
}

test "parallel shader precompile - empty variant" {
    if (!build_options.enable_spice) return error.SkipZigTest;

    const testCompiler = struct {
        fn compile(variant: []const SpecValue, allocator: std.mem.Allocator) ShaderCompileResult {
            _ = allocator;
            return .{
                .success = variant.len > 0,
                .variant_index = 0,
                .error_msg = if (variant.len == 0) "empty variant" else null,
            };
        }
    }.compile;

    var variants: [4][]const SpecValue = undefined;
    var results: [4]ShaderCompileResult = undefined;

    for (&variants) |*v| {
        v.* = &[_]SpecValue{}; // Empty
    }

    parallelShaderPrecompile(std.testing.allocator, &variants, &results, testCompiler);

    for (results) |result| {
        try std.testing.expect(!result.success);
    }
}

test "parallel context create - invalid device" {
    if (!build_options.enable_spice) return error.SkipZigTest;

    const testCreator = struct {
        fn create(device: DeviceInfo, allocator: std.mem.Allocator) ContextCreateResult {
            _ = allocator;
            return .{
                .success = device.physical_device != 0,
                .context_handle = if (device.physical_device != 0) device.physical_device else null,
                .error_code = if (device.physical_device == 0) @as(?i32, -1) else null,
            };
        }
    }.create;

    var devices: [4]DeviceInfo = undefined;
    var results: [4]ContextCreateResult = undefined;

    for (&devices) |*d| {
        d.* = .{
            .physical_device = 0, // Invalid
            .queue_family = 0,
            .device_name = "Invalid Device",
        };
    }

    parallelContextCreate(std.testing.allocator, &devices, &results, testCreator);

    for (results) |result| {
        try std.testing.expect(!result.success);
        try std.testing.expectEqual(@as(?i32, -1), result.error_code);
    }
}

test "parallel command record" {
    if (!build_options.enable_spice) return error.SkipZigTest;

    const testRecorder = struct {
        fn record(input: CommandBufferInput) CommandBufferResult {
            return .{
                .success = input.buffer_handle != 0,
                .error_code = null,
            };
        }
    }.record;

    var inputs: [8]CommandBufferInput = undefined;
    var results: [8]CommandBufferResult = undefined;

    for (&inputs, 0..) |*input, i| {
        input.* = .{
            .buffer_handle = @intCast(i + 1),
            .dispatch_x = 32,
            .dispatch_y = 32,
            .dispatch_z = 1,
            .pipeline = 100,
        };
    }

    parallelCommandRecord(&inputs, &results, testRecorder);

    for (results) |result| {
        try std.testing.expect(result.success);
    }
}

test "parallel buffer copy" {
    if (!build_options.enable_spice) return error.SkipZigTest;

    const testCopier = struct {
        fn copy(op: BufferCopyOp) BufferCopyResult {
            return .{
                .success = op.size > 0,
                .bytes_copied = op.size,
            };
        }
    }.copy;

    var ops: [8]BufferCopyOp = undefined;
    var results: [8]BufferCopyResult = undefined;

    for (&ops, 0..) |*op, i| {
        op.* = .{
            .src_buffer = @intCast(i),
            .dst_buffer = @intCast(i + 100),
            .src_offset = 0,
            .dst_offset = 0,
            .size = 1024,
        };
    }

    parallelBufferCopy(&ops, &results, testCopier);

    for (results) |result| {
        try std.testing.expect(result.success);
        try std.testing.expectEqual(@as(u64, 1024), result.bytes_copied);
    }
}

test "parallel buffer copy - zero size" {
    if (!build_options.enable_spice) return error.SkipZigTest;

    const testCopier = struct {
        fn copy(op: BufferCopyOp) BufferCopyResult {
            return .{
                .success = op.size > 0,
                .bytes_copied = op.size,
            };
        }
    }.copy;

    var ops: [4]BufferCopyOp = undefined;
    var results: [4]BufferCopyResult = undefined;

    for (&ops) |*op| {
        op.* = .{
            .src_buffer = 1,
            .dst_buffer = 2,
            .src_offset = 0,
            .dst_offset = 0,
            .size = 0, // Zero size
        };
    }

    parallelBufferCopy(&ops, &results, testCopier);

    for (results) |result| {
        try std.testing.expect(!result.success);
    }
}
