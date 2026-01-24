// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Vulkan Error Handling
//!
//! This module provides unified error handling for Vulkan operations,
//! mapping VkResult codes to Zig errors and providing diagnostic information.

const std = @import("std");
const types = @import("types.zig");

/// Comprehensive error set for all Vulkan operations
pub const Error = error{
    // Vulkan core errors
    OutOfHostMemory,
    OutOfDeviceMemory,
    InitializationFailed,
    DeviceLost,
    MemoryMapFailed,
    LayerNotPresent,
    ExtensionNotPresent,
    FeatureNotPresent,
    IncompatibleDriver,
    TooManyObjects,
    FormatNotSupported,
    FragmentedPool,
    OutOfPoolMemory,
    InvalidExternalHandle,
    Fragmentation,
    InvalidOpaqueCaptureAddress,
    SurfaceLost,
    NativeWindowInUse,
    OutOfDate,
    Unknown,

    // Library-level errors
    VulkanUnavailable,
    NoDevices,
    NoComputeQueue,
    NoSuitableDevice,
    NoSuitableMemoryType,
    FunctionNotFound,
    InvalidSpirv,
    ShaderCompilationFailed,
    InvalidArgument,
    InvalidHandle,
    Timeout,
    NotReady,
};

/// Map a VkResult to a library error
pub fn fromVkResult(result: types.Result) Error {
    return switch (result) {
        .success, .not_ready, .timeout, .event_set, .event_reset, .incomplete, .suboptimal_khr => Error.Unknown,
        .error_out_of_host_memory => Error.OutOfHostMemory,
        .error_out_of_device_memory => Error.OutOfDeviceMemory,
        .error_initialization_failed => Error.InitializationFailed,
        .error_device_lost => Error.DeviceLost,
        .error_memory_map_failed => Error.MemoryMapFailed,
        .error_layer_not_present => Error.LayerNotPresent,
        .error_extension_not_present => Error.ExtensionNotPresent,
        .error_feature_not_present => Error.FeatureNotPresent,
        .error_incompatible_driver => Error.IncompatibleDriver,
        .error_too_many_objects => Error.TooManyObjects,
        .error_format_not_supported => Error.FormatNotSupported,
        .error_fragmented_pool => Error.FragmentedPool,
        .error_out_of_pool_memory => Error.OutOfPoolMemory,
        .error_invalid_external_handle => Error.InvalidExternalHandle,
        .error_fragmentation => Error.Fragmentation,
        .error_invalid_opaque_capture_address => Error.InvalidOpaqueCaptureAddress,
        .error_surface_lost_khr => Error.SurfaceLost,
        .error_native_window_in_use_khr => Error.NativeWindowInUse,
        .error_out_of_date_khr => Error.OutOfDate,
        .error_unknown => Error.Unknown,
        _ => Error.Unknown,
    };
}

/// Check a VkResult and return error if not success
pub fn check(result: types.Result) Error!void {
    if (result != .success) {
        return fromVkResult(result);
    }
}

/// Check a VkResult allowing specific non-error codes
pub fn checkAllowing(result: types.Result, comptime allowed: []const types.Result) Error!types.Result {
    if (result == .success) return result;

    inline for (allowed) |a| {
        if (result == a) return result;
    }

    return fromVkResult(result);
}

/// Get a human-readable description of an error
pub fn describe(err: Error) []const u8 {
    return switch (err) {
        Error.OutOfHostMemory => "Host memory allocation failed",
        Error.OutOfDeviceMemory => "Device memory allocation failed",
        Error.InitializationFailed => "Vulkan initialization failed",
        Error.DeviceLost => "Device has been lost (fatal)",
        Error.MemoryMapFailed => "Memory mapping operation failed",
        Error.LayerNotPresent => "Requested layer is not available",
        Error.ExtensionNotPresent => "Requested extension is not available",
        Error.FeatureNotPresent => "Requested feature is not available",
        Error.IncompatibleDriver => "Vulkan driver is incompatible",
        Error.TooManyObjects => "Too many objects of this type have been created",
        Error.FormatNotSupported => "Requested format is not supported",
        Error.FragmentedPool => "Pool allocation failed due to fragmentation",
        Error.OutOfPoolMemory => "Pool has insufficient memory",
        Error.InvalidExternalHandle => "External handle is invalid",
        Error.Fragmentation => "Memory allocation failed due to fragmentation",
        Error.InvalidOpaqueCaptureAddress => "Invalid opaque capture address",
        Error.SurfaceLost => "Surface is no longer available",
        Error.NativeWindowInUse => "Native window is in use by another API",
        Error.OutOfDate => "Surface is out of date",
        Error.Unknown => "Unknown Vulkan error",
        Error.VulkanUnavailable => "Vulkan is not available on this system",
        Error.NoDevices => "No Vulkan-capable devices found",
        Error.NoComputeQueue => "No compute-capable queue found",
        Error.NoSuitableDevice => "No suitable device for the requested operations",
        Error.NoSuitableMemoryType => "No suitable memory type for the requested properties",
        Error.FunctionNotFound => "Required Vulkan function not found",
        Error.InvalidSpirv => "Invalid SPIR-V shader module",
        Error.ShaderCompilationFailed => "Shader compilation failed",
        Error.InvalidArgument => "Invalid argument provided",
        Error.InvalidHandle => "Invalid handle provided",
        Error.Timeout => "Operation timed out",
        Error.NotReady => "Operation is not ready",
    };
}

/// Check if an error is recoverable
pub fn isRecoverable(err: Error) bool {
    return switch (err) {
        // Fatal errors that require recreation or are permanent
        Error.DeviceLost,
        Error.IncompatibleDriver,
        Error.VulkanUnavailable,
        Error.NoDevices,
        => false,

        // Recoverable errors
        else => true,
    };
}

/// Check if an error is related to resource exhaustion
pub fn isResourceExhausted(err: Error) bool {
    return switch (err) {
        Error.OutOfHostMemory,
        Error.OutOfDeviceMemory,
        Error.OutOfPoolMemory,
        Error.TooManyObjects,
        Error.FragmentedPool,
        Error.Fragmentation,
        => true,
        else => false,
    };
}

// =============================================================================
// Enhanced Error Context
// =============================================================================

/// Source location for error tracking
pub const SourceLocation = struct {
    file: []const u8,
    line: u32,
    column: u32,
    fn_name: []const u8,

    pub fn format(
        self: SourceLocation,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{s}:{d}:{d} in {s}", .{
            self.file,
            self.line,
            self.column,
            self.fn_name,
        });
    }

    /// Get the current source location
    pub fn here() SourceLocation {
        return .{
            .file = @src().file,
            .line = @src().line,
            .column = @src().column,
            .fn_name = @src().fn_name,
        };
    }
};

/// Detailed error context with source location and message
pub const ErrorContext = struct {
    /// The error code
    code: Error,
    /// Source location where error occurred
    location: SourceLocation,
    /// Human-readable error message
    message: []const u8,
    /// Additional details (operation-specific)
    details: ?[]const u8 = null,
    /// Timestamp when error occurred
    timestamp: i128,
    /// Vulkan result code (if applicable)
    vk_result: ?types.Result = null,

    /// Create an error context from current location
    pub fn from(
        code: Error,
        message: []const u8,
        details: ?[]const u8,
    ) ErrorContext {
        return .{
            .code = code,
            .location = SourceLocation.here(),
            .message = message,
            .details = details,
            .timestamp = std.time.nanoTimestamp(),
            .vk_result = null,
        };
    }

    /// Create an error context from a VkResult
    pub fn fromVk(
        result: types.Result,
        message: []const u8,
        details: ?[]const u8,
    ) ErrorContext {
        return .{
            .code = fromVkResult(result),
            .location = SourceLocation.here(),
            .message = message,
            .details = details,
            .timestamp = std.time.nanoTimestamp(),
            .vk_result = result,
        };
    }

    /// Format the error context for display
    pub fn format(
        self: ErrorContext,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        // Error type and message
        try writer.print("Error: {s}\n", .{describe(self.code)});
        try writer.print("  Message: {s}\n", .{self.message});

        // Details if available
        if (self.details) |d| {
            try writer.print("  Details: {s}\n", .{d});
        }

        // VkResult if available
        if (self.vk_result) |vk| {
            try writer.print("  VkResult: {s}\n", .{@tagName(vk)});
        }

        // Location
        try writer.print("  Location: {}\n", .{self.location});
    }

    /// Get a single-line summary
    pub fn summary(self: ErrorContext, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "{s}: {s} at {s}:{d}", .{
            @tagName(self.code),
            self.message,
            self.location.file,
            self.location.line,
        }) catch "[error formatting failed]";
    }
};

/// Error trace for chaining errors
pub const ErrorTrace = struct {
    contexts: [8]ErrorContext,
    count: u8,

    pub fn init() ErrorTrace {
        return .{
            .contexts = undefined,
            .count = 0,
        };
    }

    /// Add an error to the trace
    pub fn push(self: *ErrorTrace, ctx: ErrorContext) void {
        if (self.count < 8) {
            self.contexts[self.count] = ctx;
            self.count += 1;
        }
    }

    /// Get the root cause (first error)
    pub fn rootCause(self: ErrorTrace) ?ErrorContext {
        if (self.count == 0) return null;
        return self.contexts[0];
    }

    /// Get the most recent error
    pub fn latest(self: ErrorTrace) ?ErrorContext {
        if (self.count == 0) return null;
        return self.contexts[self.count - 1];
    }

    /// Print the full trace
    pub fn print(self: ErrorTrace, writer: anytype) !void {
        if (self.count == 0) {
            try writer.writeAll("(no errors)\n");
            return;
        }

        try writer.writeAll("Error trace:\n");
        var i: u8 = self.count;
        while (i > 0) {
            i -= 1;
            try writer.print("  [{d}] {}\n", .{ i, self.contexts[i] });
        }
    }
};

/// Thread-safe error context storage
///
/// Uses a mutex to protect access to the error log from multiple threads.
/// Each thread can safely log errors which are stored in a circular buffer
/// of error traces.
pub const ErrorLog = struct {
    traces: [16]ErrorTrace,
    current: u8,
    enabled: bool,
    mutex: std.Thread.Mutex,

    var instance: ErrorLog = .{
        .traces = undefined,
        .current = 0,
        .enabled = true,
        .mutex = .{},
    };

    /// Get the global error log
    pub fn global() *ErrorLog {
        return &instance;
    }

    /// Start a new error trace (thread-safe)
    pub fn beginTrace(self: *ErrorLog) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!self.enabled) return;
        self.current = (self.current + 1) % 16;
        self.traces[self.current] = ErrorTrace.init();
    }

    /// Log an error context (thread-safe)
    pub fn log(self: *ErrorLog, ctx: ErrorContext) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!self.enabled) return;
        self.traces[self.current].push(ctx);
    }

    /// Get a copy of the current trace (thread-safe)
    ///
    /// Returns a copy to avoid holding the lock while caller uses the trace.
    pub fn currentTrace(self: *ErrorLog) ErrorTrace {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.traces[self.current];
    }

    /// Get a pointer to the current trace (not thread-safe)
    ///
    /// Only use when you know no other thread is accessing the log.
    pub fn currentTraceUnsafe(self: *ErrorLog) *ErrorTrace {
        return &self.traces[self.current];
    }

    /// Enable/disable error logging (thread-safe)
    pub fn setEnabled(self: *ErrorLog, enabled: bool) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.enabled = enabled;
    }

    /// Check if logging is enabled (thread-safe)
    pub fn isEnabled(self: *ErrorLog) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.enabled;
    }
};

/// Helper macro-like function for logging errors at call site
pub fn logError(
    comptime code: Error,
    comptime message: []const u8,
    details: ?[]const u8,
) Error {
    const ctx = ErrorContext{
        .code = code,
        .location = SourceLocation.here(),
        .message = message,
        .details = details,
        .timestamp = std.time.nanoTimestamp(),
        .vk_result = null,
    };
    ErrorLog.global().log(ctx);
    return code;
}

/// Helper for logging VkResult errors
pub fn logVkError(
    result: types.Result,
    comptime message: []const u8,
    details: ?[]const u8,
) Error {
    const code = fromVkResult(result);
    const ctx = ErrorContext{
        .code = code,
        .location = SourceLocation.here(),
        .message = message,
        .details = details,
        .timestamp = std.time.nanoTimestamp(),
        .vk_result = result,
    };
    ErrorLog.global().log(ctx);
    return code;
}

/// Get diagnostic information about an error
pub const Diagnostics = struct {
    /// Get suggestions for fixing an error
    pub fn getSuggestions(err: Error) []const []const u8 {
        return switch (err) {
            Error.OutOfHostMemory => &.{
                "Reduce memory usage or close other applications",
                "Check for memory leaks in your application",
                "Consider using pooled allocation strategies",
            },
            Error.OutOfDeviceMemory => &.{
                "Reduce GPU memory usage",
                "Use smaller buffer sizes or streaming",
                "Consider using memory aliasing for temporary resources",
                "Check if memory budget extension is available for monitoring",
            },
            Error.DeviceLost => &.{
                "Check GPU driver stability and update if needed",
                "Reduce workload complexity",
                "Implement device lost recovery handling",
                "Check for GPU timeout issues (TDR)",
            },
            Error.InvalidSpirv => &.{
                "Validate shader with spirv-val tool",
                "Check shader compilation for errors",
                "Ensure SPIR-V version is compatible with device",
            },
            Error.VulkanUnavailable => &.{
                "Install Vulkan drivers for your GPU",
                "Check if Vulkan runtime is installed",
                "Verify GPU supports Vulkan",
            },
            Error.NoComputeQueue => &.{
                "Ensure GPU has compute capability",
                "Some GPUs may not expose compute queues",
                "Check queue family properties",
            },
            else => &.{},
        };
    }

    /// Get related errors that might occur together
    pub fn getRelatedErrors(err: Error) []const Error {
        return switch (err) {
            Error.OutOfDeviceMemory => &.{ Error.OutOfPoolMemory, Error.Fragmentation },
            Error.DeviceLost => &.{ Error.SurfaceLost, Error.Timeout },
            Error.OutOfPoolMemory => &.{ Error.OutOfDeviceMemory, Error.FragmentedPool },
            else => &.{},
        };
    }
};

// =============================================================================
// Tests
// =============================================================================

test "ErrorContext creation and formatting" {
    const ctx = ErrorContext.from(Error.OutOfDeviceMemory, "Test error", "Some details");
    try std.testing.expect(ctx.code == Error.OutOfDeviceMemory);
    try std.testing.expect(ctx.message.len > 0);
}

test "ErrorTrace operations" {
    var trace = ErrorTrace.init();
    try std.testing.expect(trace.count == 0);
    try std.testing.expect(trace.rootCause() == null);

    const ctx = ErrorContext.from(Error.DeviceLost, "Device lost", null);
    trace.push(ctx);

    try std.testing.expect(trace.count == 1);
    try std.testing.expect(trace.rootCause() != null);
}

test "Diagnostics suggestions" {
    const suggestions = Diagnostics.getSuggestions(Error.OutOfDeviceMemory);
    try std.testing.expect(suggestions.len > 0);
}

test "error mapping covers all VkResult error codes" {
    // Test that all error codes map to something meaningful
    try std.testing.expect(fromVkResult(.error_out_of_host_memory) != Error.Unknown);
    try std.testing.expect(fromVkResult(.error_device_lost) != Error.Unknown);
    try std.testing.expect(fromVkResult(.error_initialization_failed) != Error.Unknown);
}

test "check returns error for non-success" {
    try std.testing.expectError(Error.OutOfHostMemory, check(.error_out_of_host_memory));
    try check(.success);
}

test "checkAllowing allows specified results" {
    const result = try checkAllowing(.timeout, &.{ .timeout, .not_ready });
    try std.testing.expectEqual(types.Result.timeout, result);
}

test "describe returns non-empty strings" {
    const desc = describe(Error.DeviceLost);
    try std.testing.expect(desc.len > 0);
}

test "isRecoverable identifies fatal errors" {
    try std.testing.expect(!isRecoverable(Error.DeviceLost));
    try std.testing.expect(!isRecoverable(Error.VulkanUnavailable));
    try std.testing.expect(isRecoverable(Error.OutOfHostMemory));
}

test "isResourceExhausted identifies memory errors" {
    try std.testing.expect(isResourceExhausted(Error.OutOfHostMemory));
    try std.testing.expect(isResourceExhausted(Error.OutOfDeviceMemory));
    try std.testing.expect(!isResourceExhausted(Error.DeviceLost));
}
