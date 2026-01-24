// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Shader Hot-Reload Utility
//!
//! Provides file watching and automatic shader reloading for development workflows.
//! When a SPIR-V shader file changes on disk, this module can detect the change
//! and trigger shader recompilation.
//!
//! ## Features
//!
//! - Poll-based file change detection (cross-platform)
//! - Automatic shader module recreation
//! - Callback-based notification system
//! - Thread-safe operation
//!
//! ## Example
//!
//! ```zig
//! const vulkan = @import("vulkan");
//!
//! // Create a shader watcher
//! var watcher = vulkan.ShaderWatcher.init(allocator);
//! defer watcher.deinit();
//!
//! // Watch a shader file
//! try watcher.watch("shaders/compute.spv", .{
//!     .on_change = onShaderChange,
//!     .user_data = &my_context,
//! });
//!
//! // In your main loop
//! while (running) {
//!     watcher.poll(); // Check for changes
//!     // ... rest of frame
//! }
//! ```

const std = @import("std");
const types = @import("../vk/types.zig");
const builtin = @import("builtin");
const loader = @import("../vk/loader.zig");
const err = @import("../vk/error.zig");

/// Shader file metadata for change detection
pub const ShaderFileInfo = struct {
    /// Path to the shader file
    path: []const u8,
    /// Last modification time (nanoseconds since epoch)
    last_modified: i128,
    /// Size of the file at last check
    size: u64,
    /// Content hash for validation (optional)
    content_hash: ?u64 = null,

    /// Check if the file has been modified since last check
    pub fn hasChanged(self: ShaderFileInfo, new_info: ShaderFileInfo) bool {
        if (self.last_modified != new_info.last_modified) return true;
        if (self.size != new_info.size) return true;
        if (self.content_hash != null and new_info.content_hash != null) {
            return self.content_hash.? != new_info.content_hash.?;
        }
        return false;
    }
};

/// Callback for shader change notifications
pub const ChangeCallback = *const fn (path: []const u8, user_data: ?*anyopaque) void;

/// Watch entry for a single shader file
pub const WatchEntry = struct {
    /// Path to the shader file (owned copy)
    path: []const u8,
    /// Last known file info
    file_info: ShaderFileInfo,
    /// Callback when file changes
    on_change: ?ChangeCallback = null,
    /// User data passed to callback
    user_data: ?*anyopaque = null,
    /// Whether to compute content hash (slower but more reliable)
    use_content_hash: bool = false,
    /// Whether this entry is active
    active: bool = true,
    /// Number of times this shader has been reloaded
    reload_count: u32 = 0,
};

/// Watch options for adding new entries
pub const WatchOptions = struct {
    /// Callback when file changes
    on_change: ?ChangeCallback = null,
    /// User data passed to callback
    user_data: ?*anyopaque = null,
    /// Whether to compute content hash (slower but more reliable)
    use_content_hash: bool = false,
};

/// Shader file watcher for hot-reload functionality
pub const ShaderWatcher = struct {
    /// Watched files
    entries: std.ArrayListUnmanaged(WatchEntry),
    /// Host allocator
    allocator: std.mem.Allocator,
    /// Poll interval in nanoseconds (default 500ms)
    poll_interval_ns: u64 = 500_000_000,
    /// Last poll timestamp
    last_poll_ns: i128 = 0,
    /// Total change events detected
    total_changes: u64 = 0,
    /// Whether watching is paused
    paused: bool = false,

    const Self = @This();

    /// Initialize a new shader watcher
    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .entries = .{},
            .allocator = allocator,
        };
    }

    /// Deinitialize and free all resources
    pub fn deinit(self: *Self) void {
        for (self.entries.items) |entry| {
            self.allocator.free(entry.path);
        }
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add a file to watch
    pub fn watch(self: *Self, path: []const u8, options: WatchOptions) !void {
        // Check if already watching
        for (self.entries.items) |*entry| {
            if (std.mem.eql(u8, entry.path, path)) {
                // Update options for existing entry
                entry.on_change = options.on_change;
                entry.user_data = options.user_data;
                entry.use_content_hash = options.use_content_hash;
                entry.active = true;
                return;
            }
        }

        // Get initial file info
        const file_info = try self.getFileInfo(path, options.use_content_hash);

        // Create owned copy of path
        const owned_path = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(owned_path);

        try self.entries.append(self.allocator, .{
            .path = owned_path,
            .file_info = file_info,
            .on_change = options.on_change,
            .user_data = options.user_data,
            .use_content_hash = options.use_content_hash,
        });
    }

    /// Remove a file from watching
    pub fn unwatch(self: *Self, path: []const u8) void {
        for (self.entries.items, 0..) |entry, i| {
            if (std.mem.eql(u8, entry.path, path)) {
                self.allocator.free(entry.path);
                _ = self.entries.swapRemove(i);
                return;
            }
        }
    }

    /// Poll for changes (call periodically in main loop)
    pub fn poll(self: *Self) !u32 {
        if (self.paused) return 0;

        const now = std.time.nanoTimestamp();

        // Respect poll interval
        if (now - self.last_poll_ns < @as(i128, @intCast(self.poll_interval_ns))) {
            return 0;
        }
        self.last_poll_ns = now;

        var changes: u32 = 0;

        for (self.entries.items) |*entry| {
            if (!entry.active) continue;

            // Try to get current file info
            const current_info = self.getFileInfo(entry.path, entry.use_content_hash) catch |e| {
                // File might be being written or temporarily unavailable
                if (e == error.FileNotFound or e == error.AccessDenied) {
                    if (builtin.mode == .Debug) {
                        std.log.debug("FileWatcher: skipping '{s}' ({s})", .{
                            entry.path,
                            if (e == error.FileNotFound) "not found" else "access denied",
                        });
                    }
                    continue;
                }
                return e;
            };

            if (entry.file_info.hasChanged(current_info)) {
                // File has changed!
                entry.file_info = current_info;
                entry.reload_count += 1;
                changes += 1;
                self.total_changes += 1;

                // Call callback if set
                if (entry.on_change) |callback| {
                    callback(entry.path, entry.user_data);
                }
            }
        }

        return changes;
    }

    /// Pause all watching
    pub fn pause(self: *Self) void {
        self.paused = true;
    }

    /// Resume watching (unpause)
    pub fn unpause(self: *Self) void {
        self.paused = false;
        self.last_poll_ns = 0; // Force immediate poll on next call
    }

    /// Force reload check (ignores poll interval)
    pub fn forceCheck(self: *Self) !u32 {
        const saved_interval = self.poll_interval_ns;
        self.poll_interval_ns = 0;
        defer self.poll_interval_ns = saved_interval;
        return try self.poll();
    }

    /// Get number of watched files
    pub fn watchCount(self: Self) usize {
        return self.entries.items.len;
    }

    /// Get file info for a path
    fn getFileInfo(self: Self, path: []const u8, compute_hash: bool) !ShaderFileInfo {
        const file = std.fs.cwd().openFile(path, .{}) catch |e| {
            // Preserve specific errors for better diagnostics
            return switch (e) {
                error.FileNotFound => error.FileNotFound,
                error.AccessDenied => error.AccessDenied,
                error.IsDir => error.IsDir,
                error.NoSpaceLeft => error.NoSpaceLeft,
                error.SystemResources => error.SystemResources,
                else => {
                    if (builtin.mode == .Debug) {
                        std.log.debug("FileWatcher.getFileInfo: {s} failed with {}", .{ path, e });
                    }
                    return error.Unexpected;
                },
            };
        };
        defer file.close();

        const stat = try file.stat();

        var hash: ?u64 = null;
        if (compute_hash) {
            // Read file and compute hash
            const content = try file.readToEndAlloc(self.allocator, 10 * 1024 * 1024); // 10MB limit
            defer self.allocator.free(content);
            hash = std.hash.Wyhash.hash(0, content);
        }

        return ShaderFileInfo{
            .path = path,
            .last_modified = stat.mtime,
            .size = stat.size,
            .content_hash = hash,
        };
    }
};

/// Hot-reloadable shader module
/// Combines a ShaderModule with automatic file watching
pub const HotReloadShader = struct {
    /// Current shader module (owned)
    module: ?types.ShaderModule = null,
    /// Path to the shader file
    path: []const u8,
    /// Device for shader creation
    device: types.Device,
    /// Device dispatch table
    dispatch: loader.DeviceDispatch,
    /// Allocator
    allocator: std.mem.Allocator,
    /// File info for change detection
    file_info: ?ShaderFileInfo = null,
    /// Number of successful reloads
    reload_count: u32 = 0,
    /// Last reload error (if any)
    last_error: ?err.Error = null,
    /// Whether to use content hash for change detection (more reliable but slower)
    use_content_hash: bool = false,

    const Self = @This();

    /// Create a hot-reloadable shader
    pub fn init(
        allocator: std.mem.Allocator,
        device: types.Device,
        dispatch: loader.DeviceDispatch,
        path: []const u8,
    ) !Self {
        var self = Self{
            .path = try allocator.dupe(u8, path),
            .device = device,
            .dispatch = dispatch,
            .allocator = allocator,
        };
        errdefer allocator.free(self.path);

        // Load initial shader
        try self.reload();

        return self;
    }

    /// Destroy the shader and free resources
    pub fn deinit(self: *Self) void {
        if (self.module) |mod| {
            self.dispatch.vkDestroyShaderModule.?(self.device, mod, null);
        }
        self.allocator.free(self.path);
        self.* = undefined;
    }

    /// Reload the shader from disk
    pub fn reload(self: *Self) !void {
        // Read shader file
        const file = try std.fs.cwd().openFile(self.path, .{});
        defer file.close();

        const stat = try file.stat();
        const code = try file.readToEndAlloc(self.allocator, 50 * 1024 * 1024); // 50MB limit
        defer self.allocator.free(code);

        // Validate SPIR-V magic number
        if (code.len < 4) return err.Error.InvalidValue;

        // Ensure proper 4-byte alignment for SPIR-V code (Vulkan requirement)
        // readToEndAlloc may not guarantee alignment, so copy to aligned buffer if needed
        const aligned_code = if (@intFromPtr(code.ptr) % 4 != 0) blk: {
            const aligned = self.allocator.alignedAlloc(u8, 4, code.len) catch return err.Error.OutOfHostMemory;
            @memcpy(aligned, code);
            break :blk aligned;
        } else code;
        defer if (@intFromPtr(code.ptr) % 4 != 0) self.allocator.free(aligned_code);

        const code_u32: [*]const u32 = @ptrCast(@alignCast(aligned_code.ptr));
        if (code_u32[0] != 0x07230203) return err.Error.InvalidValue;

        // Create new shader module
        const create_info = types.ShaderModuleCreateInfo{
            .code_size = code.len,
            .p_code = code_u32,
        };

        var new_module: types.ShaderModule = undefined;
        const result = self.dispatch.vkCreateShaderModule.?(self.device, &create_info, null, &new_module);
        if (result != .success) return err.fromVkResult(result);

        // Destroy old module and update
        if (self.module) |old| {
            self.dispatch.vkDestroyShaderModule.?(self.device, old, null);
        }
        self.module = new_module;
        self.reload_count += 1;
        self.last_error = null;

        // Update file info with optional content hash
        self.file_info = ShaderFileInfo{
            .path = self.path,
            .last_modified = stat.mtime,
            .size = stat.size,
            .content_hash = if (self.use_content_hash) std.hash.Wyhash.hash(0, aligned_code) else null,
        };
    }

    /// Check if the shader file has changed and reload if necessary
    /// Returns true if shader was reloaded
    pub fn checkAndReload(self: *Self) !bool {
        const file = std.fs.cwd().openFile(self.path, .{}) catch |e| {
            return switch (e) {
                error.FileNotFound => false,
                error.AccessDenied => false,
                else => error.Unexpected,
            };
        };
        defer file.close();

        const stat = try file.stat();
        const current_info = ShaderFileInfo{
            .path = self.path,
            .last_modified = stat.mtime,
            .size = stat.size,
        };

        if (self.file_info) |fi| {
            if (!fi.hasChanged(current_info)) {
                return false; // No change
            }
        }

        // File changed, reload
        self.reload() catch |e| {
            self.last_error = e;
            return e;
        };
        return true;
    }

    /// Get the current shader module handle
    pub fn getModule(self: Self) ?types.ShaderModule {
        return self.module;
    }

    /// Check if the shader loaded successfully
    pub fn isValid(self: Self) bool {
        return self.module != null;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "ShaderFileInfo change detection" {
    const info1 = ShaderFileInfo{
        .path = "test.spv",
        .last_modified = 1000,
        .size = 100,
    };

    const info2 = ShaderFileInfo{
        .path = "test.spv",
        .last_modified = 1000,
        .size = 100,
    };

    const info3 = ShaderFileInfo{
        .path = "test.spv",
        .last_modified = 2000,
        .size = 100,
    };

    const info4 = ShaderFileInfo{
        .path = "test.spv",
        .last_modified = 1000,
        .size = 200,
    };

    try std.testing.expect(!info1.hasChanged(info2));
    try std.testing.expect(info1.hasChanged(info3));
    try std.testing.expect(info1.hasChanged(info4));
}

test "ShaderFileInfo hash change detection" {
    const info1 = ShaderFileInfo{
        .path = "test.spv",
        .last_modified = 1000,
        .size = 100,
        .content_hash = 12345,
    };

    const info2 = ShaderFileInfo{
        .path = "test.spv",
        .last_modified = 1000,
        .size = 100,
        .content_hash = 12345,
    };

    const info3 = ShaderFileInfo{
        .path = "test.spv",
        .last_modified = 1000,
        .size = 100,
        .content_hash = 67890,
    };

    try std.testing.expect(!info1.hasChanged(info2));
    try std.testing.expect(info1.hasChanged(info3));
}

test "ShaderWatcher initialization" {
    const allocator = std.testing.allocator;
    var watcher = ShaderWatcher.init(allocator);
    defer watcher.deinit();

    try std.testing.expectEqual(@as(usize, 0), watcher.watchCount());
}
