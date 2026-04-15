// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Parallel Vulkan operations using thread-pool batched submission.
//!
//! Provides high-performance parallel implementations for CPU-side
//! Vulkan operations including shader variant precompilation,
//! multi-device context creation, and batch command preparation.
//!
//! ## Overview
//!
//! This module enables parallel processing of Vulkan CPU-side operations
//! using the devnw thread-pool. Each parallel function fans out independent
//! work items as a flat batch of tasks (up to MAX_CHUNKS chunks) via
//! submitBatch + WaitGroup, then waits for completion.  If the pool is not
//! available or submission fails the operation falls back to sequential
//! execution on the caller's thread.
//!
//! ## Build Configuration
//!
//! Enable: `-Dthread_pool=true` (default)
//! Disable: `-Dthread_pool=false`
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
const tp = @import("thread_pool");

/// Maximum number of parallel chunks any fan-out operation will create.
const MAX_CHUNKS: usize = 64;

// ============================================================================
// Global Thread Pool
// ============================================================================

var global_pool: ?tp.ThreadPool = null;
var pool_mutex: std.Thread.Mutex = .{};

pub fn getPool() *tp.ThreadPool {
    pool_mutex.lock();
    defer pool_mutex.unlock();

    if (global_pool == null) {
        global_pool = tp.ThreadPool.init(std.heap.page_allocator, .{}) catch
            @panic("thread-pool: failed to init global pool");
    }
    return &global_pool.?;
}

pub fn deinitPool() void {
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

/// Per-chunk context for parallel shader compilation
const ShaderChunkCtx = struct {
    variants: []const []const SpecValue,
    results: []ShaderCompileResult,
    compiler: ShaderCompiler,
    allocator: std.mem.Allocator,
};

fn shaderChunkTask(raw: *anyopaque, _: *tp.TaskContext) void {
    const ctx: *ShaderChunkCtx = @ptrCast(@alignCast(raw));
    for (ctx.variants, 0..) |variant, i| {
        ctx.results[i] = ctx.compiler(variant, ctx.allocator);
    }
}

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

    if (variants.len < 2) {
        for (variants, 0..) |variant, i| {
            results[i] = compiler(variant, allocator);
        }
        return;
    }

    const n_chunks = @min(variants.len, MAX_CHUNKS);
    var chunk_ctxs: [MAX_CHUNKS]ShaderChunkCtx = undefined;
    var batch: [MAX_CHUNKS]tp.BatchTask = undefined;

    const chunk_size = variants.len / n_chunks;
    var offset: usize = 0;
    var n: usize = 0;
    while (n < n_chunks and offset < variants.len) : (n += 1) {
        const end = if (n == n_chunks - 1) variants.len else offset + chunk_size;
        chunk_ctxs[n] = .{
            .variants = variants[offset..end],
            .results = results[offset..end],
            .compiler = compiler,
            .allocator = allocator,
        };
        batch[n] = .{ .task_fn = shaderChunkTask, .ctx = &chunk_ctxs[n] };
        offset = end;
    }

    var wg = getPool().waitGroup();
    _ = getPool().submitBatch(batch[0..n], .{ .wait_group = &wg }) catch {
        // Submission failed — fall back to sequential execution.
        for (variants, 0..) |variant, i| {
            results[i] = compiler(variant, allocator);
        }
        return;
    };
    wg.wait();
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

/// Per-chunk context for parallel device context creation
const ContextChunkCtx = struct {
    devices: []const DeviceInfo,
    results: []ContextCreateResult,
    creator: ContextCreator,
    allocator: std.mem.Allocator,
};

fn contextChunkTask(raw: *anyopaque, _: *tp.TaskContext) void {
    const ctx: *ContextChunkCtx = @ptrCast(@alignCast(raw));
    for (ctx.devices, 0..) |device, i| {
        ctx.results[i] = ctx.creator(device, ctx.allocator);
    }
}

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

    if (devices.len < 2) {
        for (devices, 0..) |device, i| {
            results[i] = creator(device, allocator);
        }
        return;
    }

    const n_chunks = @min(devices.len, MAX_CHUNKS);
    var chunk_ctxs: [MAX_CHUNKS]ContextChunkCtx = undefined;
    var batch: [MAX_CHUNKS]tp.BatchTask = undefined;

    const chunk_size = devices.len / n_chunks;
    var offset: usize = 0;
    var n: usize = 0;
    while (n < n_chunks and offset < devices.len) : (n += 1) {
        const end = if (n == n_chunks - 1) devices.len else offset + chunk_size;
        chunk_ctxs[n] = .{
            .devices = devices[offset..end],
            .results = results[offset..end],
            .creator = creator,
            .allocator = allocator,
        };
        batch[n] = .{ .task_fn = contextChunkTask, .ctx = &chunk_ctxs[n] };
        offset = end;
    }

    var wg = getPool().waitGroup();
    _ = getPool().submitBatch(batch[0..n], .{ .wait_group = &wg }) catch {
        for (devices, 0..) |device, i| {
            results[i] = creator(device, allocator);
        }
        return;
    };
    wg.wait();
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

/// Per-chunk context for parallel command recording
const CommandChunkCtx = struct {
    inputs: []const CommandBufferInput,
    results: []CommandBufferResult,
    recorder: CommandRecorder,
};

fn commandChunkTask(raw: *anyopaque, _: *tp.TaskContext) void {
    const ctx: *CommandChunkCtx = @ptrCast(@alignCast(raw));
    for (ctx.inputs, 0..) |input, i| {
        ctx.results[i] = ctx.recorder(input);
    }
}

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

    if (inputs.len < 4) {
        for (inputs, 0..) |input, i| {
            results[i] = recorder(input);
        }
        return;
    }

    const n_chunks = @min(inputs.len, MAX_CHUNKS);
    var chunk_ctxs: [MAX_CHUNKS]CommandChunkCtx = undefined;
    var batch: [MAX_CHUNKS]tp.BatchTask = undefined;

    const chunk_size = inputs.len / n_chunks;
    var offset: usize = 0;
    var n: usize = 0;
    while (n < n_chunks and offset < inputs.len) : (n += 1) {
        const end = if (n == n_chunks - 1) inputs.len else offset + chunk_size;
        chunk_ctxs[n] = .{
            .inputs = inputs[offset..end],
            .results = results[offset..end],
            .recorder = recorder,
        };
        batch[n] = .{ .task_fn = commandChunkTask, .ctx = &chunk_ctxs[n] };
        offset = end;
    }

    var wg = getPool().waitGroup();
    _ = getPool().submitBatch(batch[0..n], .{ .wait_group = &wg }) catch {
        for (inputs, 0..) |input, i| {
            results[i] = recorder(input);
        }
        return;
    };
    wg.wait();
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

/// Per-chunk context for parallel buffer copies
const BufferCopyChunkCtx = struct {
    ops: []const BufferCopyOp,
    results: []BufferCopyResult,
    copier: BufferCopier,
};

fn bufferCopyChunkTask(raw: *anyopaque, _: *tp.TaskContext) void {
    const ctx: *BufferCopyChunkCtx = @ptrCast(@alignCast(raw));
    for (ctx.ops, 0..) |op, i| {
        ctx.results[i] = ctx.copier(op);
    }
}

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

    if (ops.len < 4) {
        for (ops, 0..) |op, i| {
            results[i] = copier(op);
        }
        return;
    }

    const n_chunks = @min(ops.len, MAX_CHUNKS);
    var chunk_ctxs: [MAX_CHUNKS]BufferCopyChunkCtx = undefined;
    var batch: [MAX_CHUNKS]tp.BatchTask = undefined;

    const chunk_size = ops.len / n_chunks;
    var offset: usize = 0;
    var n: usize = 0;
    while (n < n_chunks and offset < ops.len) : (n += 1) {
        const end = if (n == n_chunks - 1) ops.len else offset + chunk_size;
        chunk_ctxs[n] = .{
            .ops = ops[offset..end],
            .results = results[offset..end],
            .copier = copier,
        };
        batch[n] = .{ .task_fn = bufferCopyChunkTask, .ctx = &chunk_ctxs[n] };
        offset = end;
    }

    var wg = getPool().waitGroup();
    _ = getPool().submitBatch(batch[0..n], .{ .wait_group = &wg }) catch {
        for (ops, 0..) |op, i| {
            results[i] = copier(op);
        }
        return;
    };
    wg.wait();
}

// ============================================================================
// Tests
// ============================================================================

test "parallel shader precompile" {
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
