// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Comprehensive Error Recovery
//!
//! Provides utilities for handling and recovering from Vulkan errors:
//! - Automatic retry with exponential backoff
//! - Resource recreation on device lost
//! - Graceful degradation strategies
//! - Error context and debugging
//!
//! ## Usage
//!
//! ```zig
//! const recovery = @import("vulkan").recovery;
//!
//! // Retry an operation with exponential backoff
//! const result = try recovery.retry(3, .{ .base_delay_ms = 10 }, fn() !T {
//!     return riskyOperation();
//! });
//!
//! // Handle device lost with recreation
//! var handler = recovery.DeviceLostHandler.init(&ctx);
//! try handler.execute(fn(ctx: *Context) !void {
//!     // ... operations that might cause device lost
//! });
//! ```

const std = @import("std");
const builtin = @import("builtin");
const err = @import("../vk/error.zig");

/// Error severity levels
pub const ErrorSeverity = enum {
    /// Recoverable with retry
    transient,
    /// Recoverable with resource recreation
    recoverable,
    /// Fatal, cannot recover
    fatal,
    /// Unknown severity
    unknown,
};

/// Error context for debugging
pub const ErrorContext = struct {
    /// The error that occurred
    error_code: err.Error,
    /// Where the error occurred
    location: []const u8,
    /// Additional context message
    message: []const u8,
    /// Timestamp (nanoseconds since epoch)
    timestamp: i128,
    /// Retry count (if retrying)
    retry_count: u32,

    pub fn format(
        self: ErrorContext,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("[{s}] {s}: {s} (retries: {d})", .{
            self.location,
            @tagName(self.error_code),
            self.message,
            self.retry_count,
        });
    }
};

/// Retry configuration
pub const RetryConfig = struct {
    /// Maximum number of retry attempts
    max_retries: u32 = 3,
    /// Base delay between retries in milliseconds
    base_delay_ms: u64 = 10,
    /// Maximum delay between retries in milliseconds
    max_delay_ms: u64 = 5000,
    /// Exponential backoff multiplier
    backoff_multiplier: f32 = 2.0,
    /// Add jitter to delays
    add_jitter: bool = true,
    /// Only retry transient errors
    transient_only: bool = true,
};

/// Get the severity of an error
pub fn getErrorSeverity(error_code: err.Error) ErrorSeverity {
    return switch (error_code) {
        // Transient - retry may help
        err.Error.Timeout,
        err.Error.NotReady,
        => .transient,

        // Recoverable with resource recreation
        err.Error.DeviceLost,
        err.Error.OutOfDeviceMemory,
        err.Error.SurfaceLost,
        => .recoverable,

        // Fatal - cannot recover
        err.Error.OutOfHostMemory,
        err.Error.InitializationFailed,
        err.Error.IncompatibleDriver,
        err.Error.FeatureNotPresent,
        err.Error.ExtensionNotPresent,
        err.Error.LayerNotPresent,
        err.Error.NoDevices,
        err.Error.NoComputeQueue,
        err.Error.VulkanUnavailable,
        err.Error.InvalidSpirv,
        => .fatal,

        // Unknown
        else => .unknown,
    };
}

/// Check if an error is recoverable
pub fn isRecoverable(error_code: err.Error) bool {
    const severity = getErrorSeverity(error_code);
    return severity == .transient or severity == .recoverable;
}

/// Check if an error is transient (retry may help)
pub fn isTransient(error_code: err.Error) bool {
    return getErrorSeverity(error_code) == .transient;
}

/// Retry an operation with exponential backoff
pub fn retry(
    comptime T: type,
    config: RetryConfig,
    operation: anytype,
) !T {
    var attempt: u32 = 0;
    var delay: u64 = config.base_delay_ms;

    while (true) {
        const result = operation();

        if (result) |value| {
            return value;
        } else |e| {
            attempt += 1;

            // Check if we should retry
            const should_retry = blk: {
                if (attempt > config.max_retries) break :blk false;

                // Check if error is from our error set (err.Error)
                // Use inline switch to check if the error is one we know about
                const is_known_error = switch (e) {
                    err.Error.DeviceLost,
                    err.Error.OutOfDeviceMemory,
                    err.Error.OutOfHostMemory,
                    err.Error.Timeout,
                    err.Error.NotReady,
                    => true,
                    else => false,
                };

                if (is_known_error) {
                    if (config.transient_only) {
                        break :blk isTransient(e);
                    }
                    break :blk isRecoverable(e);
                }

                // For other error types, always retry up to limit
                break :blk true;
            };

            if (!should_retry) {
                return e;
            }

            // Wait before retrying
            // Cap jitter range to u32 max to prevent overflow in intRangeAtMost
            const jitter_range = @min(delay / 4, std.math.maxInt(u32));
            if (builtin.mode == .Debug and delay / 4 > std.math.maxInt(u32)) {
                std.log.debug("Jitter range truncated from {d} to {d} (delay too large)", .{
                    delay / 4,
                    jitter_range,
                });
            }
            const actual_delay = if (config.add_jitter)
                delay + @as(u64, std.crypto.random.intRangeAtMost(u32, 0, @as(u32, @intCast(jitter_range))))
            else
                delay;

            std.Thread.sleep(actual_delay * std.time.ns_per_ms);

            // Exponential backoff
            delay = @min(
                @as(u64, @intFromFloat(@as(f32, @floatFromInt(delay)) * config.backoff_multiplier)),
                config.max_delay_ms,
            );
        }
    }
}

/// Error recovery strategy
pub const RecoveryStrategy = enum {
    /// Retry the operation
    retry,
    /// Skip and continue
    skip,
    /// Recreate resources and retry
    recreate,
    /// Fail immediately
    fail,
    /// Use fallback value/behavior
    fallback,
};

/// Recovery action result
pub const RecoveryAction = union(RecoveryStrategy) {
    retry: RetryConfig,
    skip: void,
    recreate: void,
    fail: err.Error,
    fallback: void,
};

/// Error handler callback type.
///
/// ## Ownership
/// - The callback function pointer is borrowed; caller ensures it remains valid
/// - The ErrorContext pointer is valid only for the duration of the callback
/// - Do not store references to the ErrorContext; copy data if needed
///
/// ## Thread Safety
/// Error handlers are called from the thread where the error occurred.
pub const ErrorHandler = *const fn (err.Error, *const ErrorContext) RecoveryAction;

/// Default error handler
pub fn defaultErrorHandler(error_code: err.Error, context: *const ErrorContext) RecoveryAction {
    _ = context;

    return switch (getErrorSeverity(error_code)) {
        .transient => .{ .retry = .{} },
        .recoverable => .{ .recreate = {} },
        .fatal => .{ .fail = error_code },
        .unknown => .{ .fail = error_code },
    };
}

/// Error collector for aggregating errors
pub const ErrorCollector = struct {
    errors: std.ArrayListUnmanaged(ErrorContext),
    max_errors: usize,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, max_errors: usize) Self {
        return Self{
            .errors = .{},
            .max_errors = max_errors,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.errors.deinit(self.allocator);
    }

    /// Record an error
    /// Note: This is a best-effort function. If recording fails due to
    /// memory allocation, a warning is logged but no error is returned.
    pub fn record(self: *Self, error_code: err.Error, location: []const u8, message: []const u8) void {
        if (self.errors.items.len >= self.max_errors) {
            // Remove oldest error
            _ = self.errors.orderedRemove(0);
        }

        self.errors.append(self.allocator, .{
            .error_code = error_code,
            .location = location,
            .message = message,
            .timestamp = std.time.nanoTimestamp(),
            .retry_count = 0,
        }) catch |alloc_err| {
            std.log.warn("Failed to record error {s} at {s}: {s}", .{
                @errorName(error_code),
                location,
                @errorName(alloc_err),
            });
        };
    }

    /// Get error count
    pub fn count(self: Self) usize {
        return self.errors.items.len;
    }

    /// Check if any fatal errors occurred
    pub fn hasFatalErrors(self: Self) bool {
        for (self.errors.items) |e| {
            if (getErrorSeverity(e.error_code) == .fatal) {
                return true;
            }
        }
        return false;
    }

    /// Get the most recent error
    pub fn lastError(self: Self) ?ErrorContext {
        if (self.errors.items.len == 0) return null;
        return self.errors.items[self.errors.items.len - 1];
    }

    /// Clear all errors
    pub fn clear(self: *Self) void {
        self.errors.clearRetainingCapacity();
    }

    /// Print error summary
    pub fn printSummary(self: Self, writer: anytype) !void {
        try writer.print("Error summary: {d} error(s)\n", .{self.errors.items.len});
        for (self.errors.items) |e| {
            try writer.print("  - {}\n", .{e});
        }
    }
};

/// Resource guard for automatic cleanup on error
pub fn ResourceGuard(comptime T: type) type {
    return struct {
        resource: ?T,
        cleanup_fn: *const fn (*T) void,

        const Self = @This();

        pub fn init(resource: T, cleanup_fn: *const fn (*T) void) Self {
            return Self{
                .resource = resource,
                .cleanup_fn = cleanup_fn,
            };
        }

        /// Release ownership (caller takes responsibility)
        pub fn release(self: *Self) T {
            const r = self.resource.?;
            self.resource = null;
            return r;
        }

        /// Get reference (guard keeps ownership)
        pub fn get(self: *Self) ?*T {
            if (self.resource) |*r| return r;
            return null;
        }

        /// Cleanup on scope exit
        pub fn deinit(self: *Self) void {
            if (self.resource) |*r| {
                self.cleanup_fn(r);
                self.resource = null;
            }
        }
    };
}

/// Scope guard for executing cleanup on scope exit
pub fn ScopeGuard(comptime Context: type) type {
    return struct {
        context: Context,
        cleanup_fn: *const fn (Context) void,
        active: bool,

        const Self = @This();

        pub fn init(context: Context, cleanup_fn: *const fn (Context) void) Self {
            return Self{
                .context = context,
                .cleanup_fn = cleanup_fn,
                .active = true,
            };
        }

        /// Dismiss the guard (don't execute cleanup)
        pub fn dismiss(self: *Self) void {
            self.active = false;
        }

        /// Execute cleanup
        pub fn deinit(self: *Self) void {
            if (self.active) {
                self.cleanup_fn(self.context);
            }
        }
    };
}

/// Result type with error context
pub fn ResultWithContext(comptime T: type) type {
    return union(enum) {
        ok: T,
        err: struct {
            code: err.Error,
            context: ErrorContext,
        },

        const Self = @This();

        pub fn unwrap(self: Self) !T {
            return switch (self) {
                .ok => |v| v,
                .err => |e| e.code,
            };
        }

        pub fn isOk(self: Self) bool {
            return self == .ok;
        }

        pub fn isErr(self: Self) bool {
            return self == .err;
        }
    };
}

// =============================================================================
// Device Lost Recovery
// =============================================================================

/// Resource recreation callback for device lost recovery.
///
/// ## Ownership
/// - The callback function pointer is borrowed; must remain valid until unregistered
/// - user_data is passed through unchanged; caller owns this memory
/// - The library does NOT free user_data
///
/// ## Thread Safety
/// Recreation callbacks are called from the thread that triggered device lost recovery,
/// typically during a DeviceLostHandler.handleDeviceLost() call.
///
/// ## Error Handling
/// Return an error to indicate recreation failed. The handler will retry based on
/// its configuration and eventually return MaxAttemptsExceeded if all retries fail.
pub const ResourceRecreateCallback = *const fn (user_data: ?*anyopaque) RecreationError!void;

/// Recreation error
pub const RecreationError = error{
    /// Failed to recreate a resource
    RecreationFailed,
    /// Device cannot be recovered
    UnrecoverableDevice,
    /// Resource dependencies missing
    DependencyMissing,
    /// Maximum recreation attempts exceeded
    MaxAttemptsExceeded,
    /// Out of memory during recreation
    OutOfMemory,
};

/// Registered resource for recreation
pub const RegisteredResource = struct {
    /// User data passed to callback
    user_data: ?*anyopaque,
    /// Callback to recreate the resource
    recreate_fn: ResourceRecreateCallback,
    /// Human-readable name for debugging
    name: []const u8,
    /// Priority (higher = recreated first)
    priority: i32,
    /// Whether this resource is currently valid
    is_valid: bool,
};

/// Device lost handler configuration
pub const DeviceLostConfig = struct {
    /// Maximum number of recreation attempts per device lost event
    max_recreation_attempts: u32 = 3,
    /// Delay between recreation attempts in milliseconds
    recreation_delay_ms: u64 = 100,
    /// Whether to automatically wait for device idle before recreation
    wait_for_idle: bool = true,
    /// Log recreation progress
    enable_logging: bool = true,
    /// Maximum number of registered resources
    max_resources: usize = 256,
};

/// Device lost recovery handler
///
/// Manages automatic recovery from VK_ERROR_DEVICE_LOST by:
/// 1. Tracking resources that need recreation
/// 2. Coordinating recreation in dependency order
/// 3. Providing callbacks for resource recreation
///
/// ## Usage
///
/// ```zig
/// var handler = DeviceLostHandler.init(allocator, .{});
/// defer handler.deinit();
///
/// // Register resources for recreation
/// try handler.registerResource("my_buffer", buffer_ptr, myBufferRecreate, 100);
///
/// // Execute operation with recovery
/// const result = handler.execute(myRiskyOperation, my_context);
/// ```
pub const DeviceLostHandler = struct {
    allocator: std.mem.Allocator,
    config: DeviceLostConfig,
    resources: std.ArrayListUnmanaged(RegisteredResource),
    device_lost_count: u32,
    last_device_lost_time: i128,
    is_recovering: bool,
    error_log: ErrorCollector,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: DeviceLostConfig) Self {
        return Self{
            .allocator = allocator,
            .config = config,
            .resources = .{},
            .device_lost_count = 0,
            .last_device_lost_time = 0,
            .is_recovering = false,
            .error_log = ErrorCollector.init(allocator, 100),
        };
    }

    pub fn deinit(self: *Self) void {
        self.resources.deinit(self.allocator);
        self.error_log.deinit();
    }

    /// Register a resource for automatic recreation
    pub fn registerResource(
        self: *Self,
        name: []const u8,
        user_data: ?*anyopaque,
        recreate_fn: ResourceRecreateCallback,
        priority: i32,
    ) !void {
        if (self.resources.items.len >= self.config.max_resources) {
            return error.OutOfMemory;
        }

        try self.resources.append(self.allocator, .{
            .user_data = user_data,
            .recreate_fn = recreate_fn,
            .name = name,
            .priority = priority,
            .is_valid = true,
        });

        // Keep sorted by priority (descending)
        std.mem.sort(RegisteredResource, self.resources.items, {}, struct {
            fn lessThan(_: void, a: RegisteredResource, b: RegisteredResource) bool {
                return a.priority > b.priority;
            }
        }.lessThan);
    }

    /// Unregister a resource
    pub fn unregisterResource(self: *Self, user_data: ?*anyopaque) void {
        var i: usize = 0;
        while (i < self.resources.items.len) {
            if (self.resources.items[i].user_data == user_data) {
                _ = self.resources.orderedRemove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Mark all resources as invalid (called when device is lost)
    pub fn invalidateAll(self: *Self) void {
        for (self.resources.items) |*r| {
            r.is_valid = false;
        }
    }

    /// Handle device lost error
    pub fn handleDeviceLost(self: *Self) RecreationError!void {
        if (self.is_recovering) {
            // Already recovering, don't nest
            return;
        }

        self.is_recovering = true;
        defer self.is_recovering = false;

        self.device_lost_count += 1;
        self.last_device_lost_time = std.time.nanoTimestamp();

        if (self.config.enable_logging) {
            std.log.info("Device lost detected (occurrence #{d}), initiating recovery...", .{self.device_lost_count});
        }

        // Mark all resources as invalid
        self.invalidateAll();

        // Attempt to recreate all resources
        var attempt: u32 = 0;
        while (attempt < self.config.max_recreation_attempts) : (attempt += 1) {
            if (self.config.enable_logging) {
                std.log.info("Recreation attempt {d}/{d}...", .{ attempt + 1, self.config.max_recreation_attempts });
            }

            const success = self.recreateResources();
            if (success) {
                if (self.config.enable_logging) {
                    std.log.info("Device recovery successful after {d} attempt(s)", .{attempt + 1});
                }
                return;
            }

            // Wait before retrying
            if (attempt + 1 < self.config.max_recreation_attempts) {
                std.Thread.sleep(self.config.recreation_delay_ms * std.time.ns_per_ms);
            }
        }

        if (self.config.enable_logging) {
            std.log.err("Device recovery failed after {d} attempts", .{self.config.max_recreation_attempts});
        }

        return RecreationError.MaxAttemptsExceeded;
    }

    /// Attempt to recreate all invalid resources
    fn recreateResources(self: *Self) bool {
        var all_valid = true;

        for (self.resources.items) |*resource| {
            if (!resource.is_valid) {
                if (self.config.enable_logging) {
                    std.log.debug("Recreating resource: {s}", .{resource.name});
                }

                if (resource.recreate_fn(resource.user_data)) {
                    resource.is_valid = true;
                    if (self.config.enable_logging) {
                        std.log.debug("Successfully recreated: {s}", .{resource.name});
                    }
                } else |e| {
                    all_valid = false;
                    self.error_log.record(
                        err.Error.DeviceLost,
                        resource.name,
                        @errorName(e),
                    );
                    if (self.config.enable_logging) {
                        std.log.warn("Failed to recreate: {s}: {s}", .{ resource.name, @errorName(e) });
                    }
                }
            }
        }

        return all_valid;
    }

    /// Execute an operation with device lost recovery
    pub fn execute(
        self: *Self,
        comptime T: type,
        operation: anytype,
        context: anytype,
    ) !T {
        const result = operation(context);

        if (result) |value| {
            return value;
        } else |e| {
            // Check if this is a device lost error
            if (@TypeOf(e) == err.Error and e == err.Error.DeviceLost) {
                try self.handleDeviceLost();
                // Retry the operation after recovery
                return operation(context);
            }
            return e;
        }
    }

    /// Get recovery statistics
    pub fn getStats(self: Self) DeviceLostStats {
        var valid_count: usize = 0;
        for (self.resources.items) |r| {
            if (r.is_valid) valid_count += 1;
        }

        return .{
            .device_lost_count = self.device_lost_count,
            .registered_resources = self.resources.items.len,
            .valid_resources = valid_count,
            .last_device_lost_time = self.last_device_lost_time,
            .is_recovering = self.is_recovering,
        };
    }
};

/// Device lost recovery statistics
pub const DeviceLostStats = struct {
    device_lost_count: u32,
    registered_resources: usize,
    valid_resources: usize,
    last_device_lost_time: i128,
    is_recovering: bool,
};

/// Wrapper for executing operations with automatic device lost recovery
pub fn withDeviceRecovery(
    handler: *DeviceLostHandler,
    comptime T: type,
    operation: anytype,
    context: anytype,
) !T {
    return handler.execute(T, operation, context);
}

// =============================================================================
// Automatic Device Lost Detection
// =============================================================================

/// Result checker that automatically triggers device lost recovery
///
/// Use this to wrap Vulkan operations that may return VK_ERROR_DEVICE_LOST:
///
/// ```zig
/// var checker = DeviceLostChecker.init(&handler);
///
/// // Instead of: try vk.check(result);
/// // Use:
/// try checker.check(result);  // Will trigger recovery on device lost
/// ```
pub const DeviceLostChecker = struct {
    handler: *DeviceLostHandler,
    auto_recover: bool = true,

    const Self = @This();

    pub fn init(handler: *DeviceLostHandler) Self {
        return .{ .handler = handler };
    }

    /// Check a VkResult and trigger recovery if device lost
    pub fn check(self: *Self, result: anytype) !void {
        // Handle raw VkResult values
        if (@TypeOf(result) == i32) {
            if (result == -4) { // VK_ERROR_DEVICE_LOST
                if (self.auto_recover) {
                    try self.handler.handleDeviceLost();
                }
                return err.Error.DeviceLost;
            }
            if (result != 0) { // VK_SUCCESS
                return err.fromVkResult(@enumFromInt(result));
            }
            return;
        }

        // Handle typed Result enum
        const Result = @TypeOf(result);
        if (@hasField(Result, "error_device_lost")) {
            if (result == .error_device_lost) {
                if (self.auto_recover) {
                    try self.handler.handleDeviceLost();
                }
                return err.Error.DeviceLost;
            }
        }
    }

    /// Check and return the error (for use with catch)
    /// Note: If device lost recovery fails, the recovery error is logged and
    /// the original DeviceLost error is still returned.
    pub fn checkCatch(self: *Self, result: anytype) err.Error!void {
        return self.check(result) catch |e| {
            if (e == err.Error.DeviceLost and self.auto_recover) {
                self.handler.handleDeviceLost() catch |recovery_err| {
                    std.log.err("Device lost recovery failed: {s}", .{@errorName(recovery_err)});
                    // Record the failure in the error log for later analysis
                    self.handler.error_log.record(
                        err.Error.DeviceLost,
                        "DeviceLostChecker.checkCatch",
                        @errorName(recovery_err),
                    );
                };
            }
            return e;
        };
    }
};

/// Global device lost handler for simple use cases
///
/// For applications that don't need complex resource management,
/// this provides a simple global handler.
pub const GlobalDeviceLostHandler = struct {
    var instance: ?DeviceLostHandler = null;
    var mutex: std.Thread.Mutex = .{};

    /// Initialize the global handler (call once at startup)
    pub fn init(allocator: std.mem.Allocator, config: DeviceLostConfig) void {
        mutex.lock();
        defer mutex.unlock();
        if (instance == null) {
            instance = DeviceLostHandler.init(allocator, config);
        }
    }

    /// Deinitialize the global handler
    pub fn deinit() void {
        mutex.lock();
        defer mutex.unlock();
        if (instance) |*h| {
            h.deinit();
            instance = null;
        }
    }

    /// Get the global handler (returns null if not initialized)
    pub fn get() ?*DeviceLostHandler {
        mutex.lock();
        defer mutex.unlock();
        if (instance) |*h| return h;
        return null;
    }

    /// Register a resource with the global handler
    pub fn registerResource(
        name: []const u8,
        user_data: ?*anyopaque,
        recreate_fn: ResourceRecreateCallback,
        priority: i32,
    ) !void {
        mutex.lock();
        defer mutex.unlock();
        if (instance) |*h| {
            try h.registerResource(name, user_data, recreate_fn, priority);
        }
    }

    /// Handle device lost on the global handler
    pub fn handleDeviceLost() RecreationError!void {
        mutex.lock();
        defer mutex.unlock();
        if (instance) |*h| {
            return h.handleDeviceLost();
        }
    }
};

/// Execute an operation with automatic device lost recovery
///
/// This is the simplest way to add device lost recovery to existing code:
///
/// ```zig
/// // Before:
/// const result = try myVulkanOperation();
///
/// // After:
/// const result = try withDeviceRecoverySimple(struct {
///     pub fn call() !T {
///         return myVulkanOperation();
///     }
/// }.call);
/// ```
pub fn withDeviceRecoverySimple(
    comptime T: type,
    operation: *const fn () anyerror!T,
) anyerror!T {
    var attempts: u32 = 0;
    const max_attempts: u32 = 3;

    while (attempts < max_attempts) : (attempts += 1) {
        const result = operation();

        if (result) |value| {
            return value;
        } else |e| {
            // Check if this is a device lost error
            if (e == err.Error.DeviceLost) {
                // Try global handler first
                if (GlobalDeviceLostHandler.get()) |h| {
                    h.handleDeviceLost() catch |recovery_err| {
                        std.log.warn("Device recovery failed: {s}", .{@errorName(recovery_err)});
                    };
                    // Retry after recovery
                    continue;
                }
            }
            // Not device lost or no handler, propagate error
            return e;
        }
    }

    return err.Error.DeviceLost;
}

/// Callback wrapper that handles device lost transparently
///
/// Wraps a Vulkan operation callback to automatically handle device lost:
///
/// ```zig
/// const safe_fn = wrapWithRecovery(myVulkanFunction);
/// const result = try safe_fn(args);
/// ```
pub fn wrapWithRecovery(
    comptime func: anytype,
) @TypeOf(func) {
    const Fn = @TypeOf(func);
    const info = @typeInfo(Fn);

    return struct {
        fn wrapped(args: std.meta.ArgsTuple(Fn)) callconv(info.@"fn".calling_convention) @typeInfo(Fn).@"fn".return_type {
            var attempts: u32 = 0;
            while (attempts < 3) : (attempts += 1) {
                const result = @call(.auto, func, args);
                if (result) |value| {
                    return value;
                } else |e| {
                    if (e == err.Error.DeviceLost) {
                        if (GlobalDeviceLostHandler.get()) |h| {
                            h.handleDeviceLost() catch |recovery_err| {
                                std.log.warn("Device recovery attempt {d}/3 failed: {s}", .{
                                    attempts + 1,
                                    @errorName(recovery_err),
                                });
                                continue;
                            };
                            continue;
                        }
                    }
                    return e;
                }
            }
            return err.Error.DeviceLost;
        }
    }.wrapped;
}

/// Create a resource guard that automatically unregisters on cleanup
pub fn AutoUnregister(comptime T: type) type {
    return struct {
        handler: *DeviceLostHandler,
        user_data: *T,

        const Self = @This();

        pub fn init(
            handler: *DeviceLostHandler,
            name: []const u8,
            user_data: *T,
            recreate_fn: ResourceRecreateCallback,
            priority: i32,
        ) !Self {
            try handler.registerResource(name, user_data, recreate_fn, priority);
            return Self{
                .handler = handler,
                .user_data = user_data,
            };
        }

        pub fn deinit(self: *Self) void {
            self.handler.unregisterResource(self.user_data);
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

test "DeviceLostHandler init/deinit" {
    var handler = DeviceLostHandler.init(std.testing.allocator, .{});
    defer handler.deinit();

    const stats = handler.getStats();
    try std.testing.expectEqual(@as(u32, 0), stats.device_lost_count);
    try std.testing.expectEqual(@as(usize, 0), stats.registered_resources);
}

test "DeviceLostHandler resource registration" {
    var handler = DeviceLostHandler.init(std.testing.allocator, .{});
    defer handler.deinit();

    var dummy: u32 = 42;

    try handler.registerResource("test_resource", &dummy, struct {
        fn recreate(_: ?*anyopaque) RecreationError!void {}
    }.recreate, 100);

    const stats = handler.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.registered_resources);
    try std.testing.expectEqual(@as(usize, 1), stats.valid_resources);

    handler.unregisterResource(&dummy);
    try std.testing.expectEqual(@as(usize, 0), handler.getStats().registered_resources);
}

test "DeviceLostHandler invalidateAll" {
    var handler = DeviceLostHandler.init(std.testing.allocator, .{});
    defer handler.deinit();

    var dummy1: u32 = 1;
    var dummy2: u32 = 2;

    try handler.registerResource("res1", &dummy1, struct {
        fn recreate(_: ?*anyopaque) RecreationError!void {}
    }.recreate, 100);

    try handler.registerResource("res2", &dummy2, struct {
        fn recreate(_: ?*anyopaque) RecreationError!void {}
    }.recreate, 50);

    try std.testing.expectEqual(@as(usize, 2), handler.getStats().valid_resources);

    handler.invalidateAll();

    try std.testing.expectEqual(@as(usize, 0), handler.getStats().valid_resources);
}

test "ErrorSeverity classification" {
    try std.testing.expectEqual(ErrorSeverity.transient, getErrorSeverity(err.Error.Timeout));
    try std.testing.expectEqual(ErrorSeverity.recoverable, getErrorSeverity(err.Error.DeviceLost));
    try std.testing.expectEqual(ErrorSeverity.fatal, getErrorSeverity(err.Error.OutOfHostMemory));
}

test "isRecoverable" {
    try std.testing.expect(isRecoverable(err.Error.Timeout));
    try std.testing.expect(isRecoverable(err.Error.DeviceLost));
    try std.testing.expect(!isRecoverable(err.Error.OutOfHostMemory));
}

test "isTransient" {
    try std.testing.expect(isTransient(err.Error.Timeout));
    try std.testing.expect(!isTransient(err.Error.DeviceLost));
}

test "RetryConfig defaults" {
    const config = RetryConfig{};
    try std.testing.expectEqual(@as(u32, 3), config.max_retries);
    try std.testing.expectEqual(@as(u64, 10), config.base_delay_ms);
}

test "ErrorCollector basic operations" {
    var collector = ErrorCollector.init(std.testing.allocator, 10);
    defer collector.deinit();

    try std.testing.expectEqual(@as(usize, 0), collector.count());

    collector.record(err.Error.Timeout, "test", "test message");
    try std.testing.expectEqual(@as(usize, 1), collector.count());

    try std.testing.expect(!collector.hasFatalErrors());

    collector.record(err.Error.OutOfHostMemory, "test", "fatal error");
    try std.testing.expect(collector.hasFatalErrors());
}

// =============================================================================
// Integration Tests with Mock Context
// =============================================================================

test "DeviceLostHandler full recovery cycle" {
    var handler = DeviceLostHandler.init(std.testing.allocator, .{
        .max_recreation_attempts = 3,
        .recreation_delay_ms = 1,
        .enable_logging = false,
    });
    defer handler.deinit();

    // Track recreation calls
    const RecreationState = struct {
        var recreate_count: u32 = 0;
        var should_succeed: bool = true;
    };
    RecreationState.recreate_count = 0;
    RecreationState.should_succeed = true;

    var dummy: u32 = 42;

    try handler.registerResource("buffer1", &dummy, struct {
        fn recreate(_: ?*anyopaque) RecreationError!void {
            RecreationState.recreate_count += 1;
            if (!RecreationState.should_succeed) {
                return RecreationError.RecreationFailed;
            }
        }
    }.recreate, 100);

    // Simulate device lost
    handler.invalidateAll();
    try std.testing.expectEqual(@as(usize, 0), handler.getStats().valid_resources);

    // Trigger recovery (should succeed on first attempt)
    try handler.handleDeviceLost();

    // Verify resource is now valid
    try std.testing.expectEqual(@as(usize, 1), handler.getStats().valid_resources);
    try std.testing.expectEqual(@as(u32, 1), RecreationState.recreate_count);
    try std.testing.expectEqual(@as(u32, 1), handler.device_lost_count);
}

test "DeviceLostHandler recovery with multiple resources" {
    var handler = DeviceLostHandler.init(std.testing.allocator, .{
        .max_recreation_attempts = 2,
        .recreation_delay_ms = 1,
        .enable_logging = false,
    });
    defer handler.deinit();

    const RecreationOrder = struct {
        var order: std.ArrayListUnmanaged(u32) = .{};
        var allocator: ?std.mem.Allocator = null;
    };
    RecreationOrder.allocator = std.testing.allocator;
    RecreationOrder.order = .{};
    defer RecreationOrder.order.deinit(std.testing.allocator);

    var resource1: u32 = 1;
    var resource2: u32 = 2;
    var resource3: u32 = 3;

    // Register resources with different priorities
    try handler.registerResource("low_priority", &resource3, struct {
        fn recreate(data: ?*anyopaque) RecreationError!void {
            if (RecreationOrder.allocator) |alloc| {
                const val: *u32 = @ptrCast(@alignCast(data orelse return));
                RecreationOrder.order.append(alloc, val.*) catch |e| {
                    std.log.warn("test recreate: failed to record order: {s}", .{@errorName(e)});
                };
            }
        }
    }.recreate, 10); // Low priority

    try handler.registerResource("high_priority", &resource1, struct {
        fn recreate(data: ?*anyopaque) RecreationError!void {
            if (RecreationOrder.allocator) |alloc| {
                const val: *u32 = @ptrCast(@alignCast(data orelse return));
                RecreationOrder.order.append(alloc, val.*) catch |e| {
                    std.log.warn("test recreate: failed to record order: {s}", .{@errorName(e)});
                };
            }
        }
    }.recreate, 100); // High priority

    try handler.registerResource("med_priority", &resource2, struct {
        fn recreate(data: ?*anyopaque) RecreationError!void {
            if (RecreationOrder.allocator) |alloc| {
                const val: *u32 = @ptrCast(@alignCast(data orelse return));
                RecreationOrder.order.append(alloc, val.*) catch |e| {
                    std.log.warn("test recreate: failed to record order: {s}", .{@errorName(e)});
                };
            }
        }
    }.recreate, 50); // Medium priority

    handler.invalidateAll();
    try handler.handleDeviceLost();

    // Verify recreation order (highest priority first)
    try std.testing.expectEqual(@as(usize, 3), RecreationOrder.order.items.len);
    try std.testing.expectEqual(@as(u32, 1), RecreationOrder.order.items[0]); // High (100)
    try std.testing.expectEqual(@as(u32, 2), RecreationOrder.order.items[1]); // Med (50)
    try std.testing.expectEqual(@as(u32, 3), RecreationOrder.order.items[2]); // Low (10)
}

test "DeviceLostHandler recovery failure after max attempts" {
    var handler = DeviceLostHandler.init(std.testing.allocator, .{
        .max_recreation_attempts = 2,
        .recreation_delay_ms = 1,
        .enable_logging = false,
    });
    defer handler.deinit();

    var dummy: u32 = 0;

    // Register a resource that always fails to recreate
    try handler.registerResource("failing_resource", &dummy, struct {
        fn recreate(_: ?*anyopaque) RecreationError!void {
            return RecreationError.RecreationFailed;
        }
    }.recreate, 100);

    handler.invalidateAll();

    // Should fail after max attempts
    const result = handler.handleDeviceLost();
    try std.testing.expectError(RecreationError.MaxAttemptsExceeded, result);
}

test "DeviceLostChecker auto recovery trigger" {
    var handler = DeviceLostHandler.init(std.testing.allocator, .{
        .max_recreation_attempts = 1,
        .recreation_delay_ms = 1,
        .enable_logging = false,
    });
    defer handler.deinit();

    var dummy: u32 = 0;
    try handler.registerResource("auto_recover", &dummy, struct {
        fn recreate(_: ?*anyopaque) RecreationError!void {}
    }.recreate, 100);

    var checker = DeviceLostChecker.init(&handler);

    // Check a device lost result (raw VkResult value)
    const device_lost_result: i32 = -4; // VK_ERROR_DEVICE_LOST
    const check_result = checker.check(device_lost_result);
    try std.testing.expectError(err.Error.DeviceLost, check_result);

    // Verify recovery was triggered
    try std.testing.expectEqual(@as(u32, 1), handler.device_lost_count);
}

test "ErrorCollector max capacity enforcement" {
    var collector = ErrorCollector.init(std.testing.allocator, 3);
    defer collector.deinit();

    // Add 5 errors (capacity is 3)
    collector.record(err.Error.Timeout, "loc1", "msg1");
    collector.record(err.Error.NotReady, "loc2", "msg2");
    collector.record(err.Error.DeviceLost, "loc3", "msg3");
    collector.record(err.Error.OutOfDeviceMemory, "loc4", "msg4");
    collector.record(err.Error.SurfaceLost, "loc5", "msg5");

    // Should only have 3 errors (oldest removed)
    try std.testing.expectEqual(@as(usize, 3), collector.count());

    // Most recent should be last
    const last = collector.lastError().?;
    try std.testing.expectEqual(err.Error.SurfaceLost, last.error_code);
}

test "ResourceGuard cleanup on scope exit" {
    const TestResource = struct {
        value: u32,
        cleaned: *bool,
    };

    var cleaned = false;
    const resource = TestResource{ .value = 42, .cleaned = &cleaned };

    {
        var guard = ResourceGuard(TestResource).init(resource, struct {
            fn cleanup(r: *TestResource) void {
                r.cleaned.* = true;
            }
        }.cleanup);
        defer guard.deinit();

        // Use the resource
        try std.testing.expectEqual(@as(u32, 42), guard.resource.?.value);
    }

    // Cleanup should have been called
    try std.testing.expect(cleaned);
}

test "ScopeGuard dismiss prevents cleanup" {
    const Tracker = struct {
        var cleanup_count: u32 = 0;
    };
    Tracker.cleanup_count = 0;

    {
        var guard = ScopeGuard(void).init({}, struct {
            fn cleanup(_: void) void {
                Tracker.cleanup_count += 1;
            }
        }.cleanup);
        defer guard.deinit();

        // Dismiss the guard
        guard.dismiss();
    }

    // Cleanup should NOT have been called
    try std.testing.expectEqual(@as(u32, 0), Tracker.cleanup_count);
}
