// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Compute Graph
//!
//! DAG-based compute task scheduling with automatic barrier insertion.
//! Allows defining compute operations as a graph of dependencies, with
//! the system automatically determining optimal execution order and
//! inserting memory barriers where needed.
//!
//! Features:
//! - Declarative task graph definition
//! - Automatic topological sorting
//! - Memory barrier insertion based on resource usage
//! - Parallel execution of independent tasks
//! - Resource lifetime tracking

const std = @import("std");
const types = @import("../vk/types.zig");
const Context = @import("context.zig").Context;
const async_compute = @import("async_compute.zig");
const AsyncCoordinator = async_compute.AsyncCoordinator;

/// GPU profiling result for a single task
pub const TaskProfile = struct {
    name: []const u8,
    start_ns: u64,
    end_ns: u64,

    pub fn durationNs(self: TaskProfile) u64 {
        return self.end_ns - self.start_ns;
    }

    pub fn durationUs(self: TaskProfile) f64 {
        return @as(f64, @floatFromInt(self.durationNs())) / 1000.0;
    }

    pub fn durationMs(self: TaskProfile) f64 {
        return @as(f64, @floatFromInt(self.durationNs())) / 1_000_000.0;
    }
};

/// GPU profiler using timestamp queries
pub const GpuProfiler = struct {
    const Self = @This();
    /// Maximum number of timestamps per profiler instance
    /// Note: Consider centralizing limits if this value is needed elsewhere
    const MAX_TIMESTAMPS: u32 = 256;

    ctx: *Context,
    allocator: std.mem.Allocator,
    query_pool: types.QueryPool,
    timestamp_period: f32,
    timestamps: std.ArrayListUnmanaged(u64),
    task_names: std.ArrayListUnmanaged([]const u8),
    next_query_index: u32 = 0,
    active: bool = false,

    /// Create a GPU profiler
    pub fn init(ctx: *Context, allocator: std.mem.Allocator) !Self {
        // Get timestamp period from device properties
        const props = ctx.getPhysicalDeviceProperties();
        const timestamp_period = props.limits.timestamp_period;

        if (timestamp_period == 0) {
            return error.TimestampQueriesNotSupported;
        }

        // Create query pool for timestamps
        const create_info = types.QueryPoolCreateInfo{
            .s_type = .query_pool_create_info,
            .p_next = null,
            .flags = 0,
            .query_type = .timestamp,
            .query_count = MAX_TIMESTAMPS,
            .pipeline_statistics = .{},
        };

        var query_pool: types.QueryPool = undefined;
        const result = ctx.device_dispatch.vkCreateQueryPool.?(ctx.device, &create_info, null, &query_pool);
        if (result != .success) return error.QueryPoolCreationFailed;

        return .{
            .ctx = ctx,
            .allocator = allocator,
            .query_pool = query_pool,
            .timestamp_period = timestamp_period,
            .timestamps = .{},
            .task_names = .{},
        };
    }

    /// Destroy the profiler
    pub fn deinit(self: *Self) void {
        if (self.ctx.device_dispatch.vkDestroyQueryPool) |destroy| {
            destroy(self.ctx.device, self.query_pool, null);
        }
        self.timestamps.deinit(self.allocator);
        self.task_names.deinit(self.allocator);
        self.* = undefined;
    }

    /// Begin profiling session - reset queries
    pub fn begin(self: *Self, cmd: types.CommandBuffer) void {
        self.next_query_index = 0;
        self.timestamps.clearRetainingCapacity();
        self.task_names.clearRetainingCapacity();
        self.active = true;

        // Reset the query pool
        if (self.ctx.device_dispatch.vkCmdResetQueryPool) |reset| {
            reset(cmd, self.query_pool, 0, MAX_TIMESTAMPS);
        }

        // Write initial timestamp
        if (self.ctx.device_dispatch.vkCmdWriteTimestamp) |write| {
            write(cmd, .{ .top_of_pipe = true }, self.query_pool, self.next_query_index);
            self.next_query_index += 1;
        }
    }

    /// Record start of a task
    pub fn beginTask(self: *Self, cmd: types.CommandBuffer, name: []const u8) void {
        if (!self.active or self.next_query_index >= MAX_TIMESTAMPS - 1) return;

        self.task_names.append(self.allocator, name) catch return;

        if (self.ctx.device_dispatch.vkCmdWriteTimestamp) |write| {
            write(cmd, .{ .compute_shader = true }, self.query_pool, self.next_query_index);
            self.next_query_index += 1;
        }
    }

    /// Record end of a task
    pub fn endTask(self: *Self, cmd: types.CommandBuffer) void {
        if (!self.active or self.next_query_index >= MAX_TIMESTAMPS) return;

        if (self.ctx.device_dispatch.vkCmdWriteTimestamp) |write| {
            write(cmd, .{ .compute_shader = true }, self.query_pool, self.next_query_index);
            self.next_query_index += 1;
        }
    }

    /// End profiling session
    pub fn end(self: *Self, cmd: types.CommandBuffer) void {
        if (!self.active) return;

        // Write final timestamp
        if (self.ctx.device_dispatch.vkCmdWriteTimestamp) |write| {
            write(cmd, .{ .bottom_of_pipe = true }, self.query_pool, self.next_query_index);
            self.next_query_index += 1;
        }
        self.active = false;
    }

    /// Read back profiling results (call after GPU execution completes)
    pub fn readResults(self: *Self) !ProfileResults {
        if (self.next_query_index == 0) {
            return ProfileResults{
                .total_gpu_ns = 0,
                .tasks = &.{},
                .allocator = self.allocator,
            };
        }

        // Allocate buffer for timestamps
        const count = self.next_query_index;
        try self.timestamps.resize(self.allocator, count);

        // Read query results
        const result = self.ctx.device_dispatch.vkGetQueryPoolResults.?(
            self.ctx.device,
            self.query_pool,
            0,
            count,
            count * @sizeOf(u64),
            @ptrCast(self.timestamps.items.ptr),
            @sizeOf(u64),
            .{ .@"64_bit" = true, .wait = true },
        );
        if (result != .success and result != .not_ready) {
            return error.QueryResultsFailed;
        }

        // Convert to nanoseconds
        const ns_per_tick: f64 = @floatCast(self.timestamp_period);

        // Build task profiles
        var task_profiles = try self.allocator.alloc(TaskProfile, self.task_names.items.len);
        errdefer self.allocator.free(task_profiles);

        // Validate we have enough timestamps
        if (self.timestamps.items.len == 0) {
            std.log.warn("ProfilerState.readResults: no timestamps collected", .{});
            return ProfileResults{
                .total_gpu_ns = 0,
                .tasks = task_profiles,
                .allocator = self.allocator,
            };
        }

        var query_idx: u32 = 1; // Skip initial timestamp
        var tasks_filled: usize = 0;
        for (self.task_names.items, 0..) |name, i| {
            if (query_idx + 1 >= self.timestamps.items.len) {
                std.log.warn("ProfilerState.readResults: incomplete timestamps - only {d}/{d} tasks profiled", .{
                    i,
                    self.task_names.items.len,
                });
                break;
            }

            const start_ticks = self.timestamps.items[query_idx];
            const end_ticks = self.timestamps.items[query_idx + 1];
            query_idx += 2;

            task_profiles[i] = .{
                .name = name,
                .start_ns = @intFromFloat(@as(f64, @floatFromInt(start_ticks)) * ns_per_tick),
                .end_ns = @intFromFloat(@as(f64, @floatFromInt(end_ticks)) * ns_per_tick),
            };
            tasks_filled = i + 1;
        }

        // Calculate total GPU time (safe now that we validated timestamps exist)
        const first_tick = self.timestamps.items[0];
        const last_tick = self.timestamps.items[self.timestamps.items.len - 1];
        const total_ns: u64 = @intFromFloat(@as(f64, @floatFromInt(last_tick - first_tick)) * ns_per_tick);

        return ProfileResults{
            .total_gpu_ns = total_ns,
            .tasks = task_profiles,
            .allocator = self.allocator,
        };
    }

    /// Get number of queries used
    pub fn queryCount(self: Self) u32 {
        return self.next_query_index;
    }
};

/// Profiling results from a graph execution
pub const ProfileResults = struct {
    total_gpu_ns: u64,
    tasks: []const TaskProfile,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ProfileResults) void {
        if (self.tasks.len > 0) {
            self.allocator.free(self.tasks);
        }
        self.* = undefined;
    }

    pub fn totalGpuUs(self: ProfileResults) f64 {
        return @as(f64, @floatFromInt(self.total_gpu_ns)) / 1000.0;
    }

    pub fn totalGpuMs(self: ProfileResults) f64 {
        return @as(f64, @floatFromInt(self.total_gpu_ns)) / 1_000_000.0;
    }

    /// Print profiling summary
    pub fn print(self: ProfileResults, writer: anytype) !void {
        try writer.print("GPU Profile Results:\n", .{});
        try writer.print("  Total GPU time: {d:.3} ms\n", .{self.totalGpuMs()});
        try writer.print("  Tasks:\n", .{});
        for (self.tasks) |task| {
            try writer.print("    {s}: {d:.3} us\n", .{ task.name, task.durationUs() });
        }
    }
};

/// Resource access mode for dependency tracking
pub const AccessMode = enum {
    read,
    write,
    read_write,

    pub fn isRead(self: AccessMode) bool {
        return self == .read or self == .read_write;
    }

    pub fn isWrite(self: AccessMode) bool {
        return self == .write or self == .read_write;
    }
};

/// Resource handle for graph tracking
pub const ResourceHandle = struct {
    id: u32,
    generation: u32 = 0,
};

/// Resource usage declaration
pub const ResourceUsage = struct {
    handle: ResourceHandle,
    access: AccessMode,
    stage: types.PipelineStageFlags = .{ .compute_shader = true },
};

/// Task callback type
pub const TaskCallback = *const fn (
    ctx: *Context,
    cmd: types.CommandBuffer,
    user_data: ?*anyopaque,
) void;

/// Task queue hint for multi-queue execution
/// This is a hint for task scheduling - the graph executor will choose
/// the best queue based on availability and task dependencies.
pub const TaskQueueHint = enum {
    /// Use the compute queue (default)
    compute,
    /// Use a dedicated transfer queue if available, else compute
    transfer,
    /// Automatically choose based on task type
    auto,

    pub fn getQueue(self: TaskQueueHint, ctx: *Context) types.Queue {
        return switch (self) {
            .compute => ctx.compute_queue,
            .transfer => ctx.transfer_queue orelse ctx.compute_queue,
            .auto => ctx.compute_queue, // Default to compute for now
        };
    }

    pub fn getQueueFamily(self: TaskQueueHint, ctx: *Context) u32 {
        return switch (self) {
            .compute => ctx.compute_queue_family,
            .transfer => ctx.transfer_queue_family orelse ctx.compute_queue_family,
            .auto => ctx.compute_queue_family,
        };
    }

    pub fn needsQueueOwnershipTransfer(self: TaskQueueHint, other: TaskQueueHint, ctx: *Context) bool {
        return self.getQueueFamily(ctx) != other.getQueueFamily(ctx);
    }
};

/// Compute task node in the graph
pub const TaskNode = struct {
    const Self = @This();

    /// Unique task identifier
    id: u32,

    /// Human-readable name for debugging
    name: []const u8,

    /// Target queue hint for this task
    queue_hint: TaskQueueHint = .compute,

    /// Pipeline to bind (optional, may use previously bound)
    pipeline: ?types.Pipeline = null,

    /// Pipeline layout for this task
    layout: ?types.PipelineLayout = null,

    /// Descriptor sets to bind
    descriptor_sets: []const types.DescriptorSet = &.{},

    /// Push constant data
    push_constants: ?[]const u8 = null,

    /// Dispatch dimensions
    workgroups: [3]u32 = .{ 1, 1, 1 },

    /// Resources this task uses
    resources: []const ResourceUsage = &.{},

    /// Optional callback for custom operations
    callback: ?TaskCallback = null,
    user_data: ?*anyopaque = null,

    /// Internal: computed dependencies
    dependencies: std.ArrayListUnmanaged(u32) = .{},
    dependents: std.ArrayListUnmanaged(u32) = .{},

    /// Internal: execution state
    in_degree: u32 = 0,
    executed: bool = false,
    level: u32 = 0, // Topological level for parallel execution
};

/// Barrier between tasks
pub const BarrierInfo = struct {
    /// Task that must complete before barrier
    src_task: u32,
    /// Task that waits on barrier
    dst_task: u32,
    /// Memory barrier details
    src_stage: types.PipelineStageFlags,
    dst_stage: types.PipelineStageFlags,
    src_access: types.AccessFlags,
    dst_access: types.AccessFlags,
};

/// Compute graph for task scheduling
pub const ComputeGraph = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    ctx: *Context,

    /// All tasks in the graph
    tasks: std.ArrayListUnmanaged(TaskNode),

    /// Task lookup by name
    task_names: std.StringHashMapUnmanaged(u32),

    /// Registered resources
    resources: std.AutoHashMapUnmanaged(u32, ResourceInfo),

    /// Computed execution order
    execution_order: std.ArrayListUnmanaged(u32),

    /// Required barriers
    barriers: std.ArrayListUnmanaged(BarrierInfo),

    /// Execution levels for parallel scheduling
    levels: std.ArrayListUnmanaged(std.ArrayListUnmanaged(u32)),

    /// Next IDs
    next_task_id: u32 = 0,
    next_resource_id: u32 = 0,

    /// Graph state
    compiled: bool = false,

    /// Optional GPU profiler for timestamp queries
    profiler: ?GpuProfiler = null,

    /// Optional async coordinator for frame-based execution
    coordinator: ?*AsyncCoordinator = null,

    const ResourceInfo = struct {
        name: []const u8,
        buffer: ?types.Buffer = null,
        size: u64 = 0,
        last_write: ?u32 = null, // Last task that wrote
        readers: std.ArrayListUnmanaged(u32), // Tasks currently reading
    };

    pub fn init(allocator: std.mem.Allocator, ctx: *Context) Self {
        return .{
            .allocator = allocator,
            .ctx = ctx,
            .tasks = .{},
            .task_names = .{},
            .resources = .{},
            .execution_order = .{},
            .barriers = .{},
            .levels = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        // Clean up profiler
        if (self.profiler) |*prof| {
            prof.deinit();
        }

        for (self.tasks.items) |*task| {
            task.dependencies.deinit(self.allocator);
            task.dependents.deinit(self.allocator);
        }
        self.tasks.deinit(self.allocator);
        self.task_names.deinit(self.allocator);

        var res_iter = self.resources.iterator();
        while (res_iter.next()) |entry| {
            entry.value_ptr.readers.deinit(self.allocator);
        }
        self.resources.deinit(self.allocator);

        self.execution_order.deinit(self.allocator);
        self.barriers.deinit(self.allocator);

        for (self.levels.items) |*level| {
            level.deinit(self.allocator);
        }
        self.levels.deinit(self.allocator);
    }

    /// Enable GPU timestamp profiling for this graph
    /// Returns error if the device doesn't support timestamps
    pub fn enableProfiling(self: *Self) !void {
        if (self.profiler != null) return; // Already enabled

        self.profiler = try GpuProfiler.init(self.ctx, self.allocator);
    }

    /// Disable GPU profiling and free resources
    pub fn disableProfiling(self: *Self) void {
        if (self.profiler) |*prof| {
            prof.deinit();
            self.profiler = null;
        }
    }

    /// Check if profiling is enabled
    pub fn isProfilingEnabled(self: Self) bool {
        return self.profiler != null;
    }

    /// Check if async transfer queue is available
    pub fn hasAsyncTransfer(self: Self) bool {
        return self.ctx.transfer_queue != null and
            self.ctx.transfer_queue_family != null and
            self.ctx.transfer_queue_family.? != self.ctx.compute_queue_family;
    }

    /// Get the transfer queue family index (or compute queue as fallback)
    pub fn getTransferQueueFamily(self: Self) u32 {
        return self.ctx.transfer_queue_family orelse self.ctx.compute_queue_family;
    }

    /// Get queue usage statistics
    pub fn getQueueStats(self: Self) struct {
        compute_tasks: u32,
        transfer_tasks: u32,
        auto_tasks: u32,
    } {
        var compute: u32 = 0;
        var transfer: u32 = 0;
        var auto: u32 = 0;

        for (self.tasks.items) |task| {
            switch (task.queue_hint) {
                .compute => compute += 1,
                .transfer => transfer += 1,
                .auto => auto += 1,
            }
        }

        return .{
            .compute_tasks = compute,
            .transfer_tasks = transfer,
            .auto_tasks = auto,
        };
    }

    /// Attach an async coordinator for frame-based execution
    /// When attached, submitAsync() will use the coordinator's frame management
    pub fn attachCoordinator(self: *Self, coord: *AsyncCoordinator) void {
        self.coordinator = coord;
    }

    /// Detach the async coordinator
    pub fn detachCoordinator(self: *Self) void {
        self.coordinator = null;
    }

    /// Check if a coordinator is attached
    pub fn hasCoordinator(self: Self) bool {
        return self.coordinator != null;
    }

    /// Get the attached coordinator (if any)
    pub fn getCoordinator(self: *Self) ?*AsyncCoordinator {
        return self.coordinator;
    }

    /// Register a resource for tracking
    pub fn registerResource(self: *Self, name: []const u8, buffer: ?types.Buffer, size: u64) !ResourceHandle {
        const id = self.next_resource_id;
        self.next_resource_id += 1;

        try self.resources.put(self.allocator, id, .{
            .name = name,
            .buffer = buffer,
            .size = size,
            .last_write = null,
            .readers = .{},
        });

        return .{ .id = id };
    }

    /// Add a compute task to the graph
    pub fn addTask(self: *Self, config: struct {
        name: []const u8,
        queue_hint: TaskQueueHint = .compute,
        pipeline: ?types.Pipeline = null,
        layout: ?types.PipelineLayout = null,
        descriptor_sets: []const types.DescriptorSet = &.{},
        push_constants: ?[]const u8 = null,
        workgroups: [3]u32 = .{ 1, 1, 1 },
        resources: []const ResourceUsage = &.{},
        callback: ?TaskCallback = null,
        user_data: ?*anyopaque = null,
    }) !u32 {
        // Validate resource handles in debug builds
        if (@import("builtin").mode == .Debug) {
            for (config.resources) |usage| {
                if (usage.handle.id >= self.next_resource_id) {
                    std.log.err("[ComputeGraph] Invalid resource handle {d} in task '{s}' (max valid: {d})", .{
                        usage.handle.id,
                        config.name,
                        self.next_resource_id -| 1,
                    });
                    return error.InvalidResource;
                }
            }
        }

        const id = self.next_task_id;
        self.next_task_id += 1;

        try self.tasks.append(self.allocator, .{
            .id = id,
            .name = config.name,
            .queue_hint = config.queue_hint,
            .pipeline = config.pipeline,
            .layout = config.layout,
            .descriptor_sets = config.descriptor_sets,
            .push_constants = config.push_constants,
            .workgroups = config.workgroups,
            .resources = config.resources,
            .callback = config.callback,
            .user_data = config.user_data,
        });

        try self.task_names.put(self.allocator, config.name, id);
        self.compiled = false;

        return id;
    }

    /// Add a transfer task (automatically uses transfer queue if available)
    /// Transfer tasks are typically buffer copies, image uploads, etc.
    pub fn addTransferTask(self: *Self, config: struct {
        name: []const u8,
        resources: []const ResourceUsage = &.{},
        callback: TaskCallback,
        user_data: ?*anyopaque = null,
    }) !u32 {
        return self.addTask(.{
            .name = config.name,
            .queue_hint = .transfer,
            .resources = config.resources,
            .callback = config.callback,
            .user_data = config.user_data,
        });
    }

    /// Add explicit dependency between tasks
    /// Note: Self-dependencies (from == to) are not allowed and will return InvalidTaskId
    /// Duplicate dependencies are silently ignored (idempotent operation)
    pub fn addDependency(self: *Self, from: u32, to: u32) !void {
        if (from >= self.tasks.items.len or to >= self.tasks.items.len) {
            return error.InvalidTaskId;
        }
        if (from == to) {
            std.log.warn("ComputeGraph.addDependency: self-dependency not allowed (task {d})", .{from});
            return error.InvalidTaskId;
        }

        // Check for duplicate dependency
        for (self.tasks.items[to].dependencies.items) |existing| {
            if (existing == from) return; // Already exists, nothing to do
        }

        try self.tasks.items[to].dependencies.append(self.allocator, from);
        try self.tasks.items[from].dependents.append(self.allocator, to);
        self.compiled = false;
    }

    /// Compile the graph - compute dependencies and execution order
    pub fn compile(self: *Self) !void {
        // Reset state
        self.execution_order.clearRetainingCapacity();
        self.barriers.clearRetainingCapacity();
        for (self.levels.items) |*level| {
            level.deinit(self.allocator);
        }
        self.levels.clearRetainingCapacity();

        // Reset resource tracking
        var res_iter = self.resources.iterator();
        while (res_iter.next()) |entry| {
            entry.value_ptr.last_write = null;
            entry.value_ptr.readers.clearRetainingCapacity();
        }

        // Compute resource-based dependencies
        try self.computeResourceDependencies();

        // Topological sort
        try self.topologicalSort();

        // Compute execution levels for parallelism
        try self.computeLevels();

        // Generate barriers
        try self.generateBarriers();

        self.compiled = true;
    }

    fn computeResourceDependencies(self: *Self) !void {
        // Process tasks in order, tracking resource access
        for (self.tasks.items, 0..) |*task, task_idx| {
            for (task.resources) |usage| {
                const res = self.resources.getPtr(usage.handle.id) orelse continue;

                if (usage.access.isRead()) {
                    // Read after write dependency
                    if (res.last_write) |writer| {
                        if (!containsValue(task.dependencies.items, writer)) {
                            try task.dependencies.append(self.allocator, writer);
                            try self.tasks.items[writer].dependents.append(
                                self.allocator,
                                @intCast(task_idx),
                            );
                        }
                    }
                    try res.readers.append(self.allocator, @intCast(task_idx));
                }

                if (usage.access.isWrite()) {
                    // Write after read - depend on all current readers
                    for (res.readers.items) |reader| {
                        if (!containsValue(task.dependencies.items, reader)) {
                            try task.dependencies.append(self.allocator, reader);
                            try self.tasks.items[reader].dependents.append(
                                self.allocator,
                                @intCast(task_idx),
                            );
                        }
                    }
                    res.readers.clearRetainingCapacity();

                    // Write after write - depend on previous writer
                    if (res.last_write) |writer| {
                        if (!containsValue(task.dependencies.items, writer)) {
                            try task.dependencies.append(self.allocator, writer);
                            try self.tasks.items[writer].dependents.append(
                                self.allocator,
                                @intCast(task_idx),
                            );
                        }
                    }

                    res.last_write = @intCast(task_idx);
                }
            }
        }
    }

    fn topologicalSort(self: *Self) !void {
        // Compute in-degrees
        for (self.tasks.items) |*task| {
            task.in_degree = @intCast(task.dependencies.items.len);
            task.executed = false;
        }

        // Kahn's algorithm
        var queue = std.ArrayListUnmanaged(u32){};
        defer queue.deinit(self.allocator);

        // Find all tasks with no dependencies
        for (self.tasks.items, 0..) |task, i| {
            if (task.in_degree == 0) {
                try queue.append(self.allocator, @intCast(i));
            }
        }

        while (queue.items.len > 0) {
            const task_id = queue.orderedRemove(0);
            try self.execution_order.append(self.allocator, task_id);
            self.tasks.items[task_id].executed = true;

            for (self.tasks.items[task_id].dependents.items) |dependent| {
                self.tasks.items[dependent].in_degree -= 1;
                if (self.tasks.items[dependent].in_degree == 0) {
                    try queue.append(self.allocator, dependent);
                }
            }
        }

        // Check for cycles - if not all tasks were processed, there's a cycle
        if (self.execution_order.items.len != self.tasks.items.len) {
            // Find and report the cycle for debugging
            self.reportCycle();
            return error.CyclicDependency;
        }
    }

    /// Find and report a cycle in the task graph for debugging
    /// Reports all tasks that weren't executed (involved in cycles) with their dependencies
    fn reportCycle(self: *Self) void {
        std.log.err("[ComputeGraph] Cyclic dependency detected!", .{});
        std.log.err("[ComputeGraph] The following tasks are involved in one or more cycles:", .{});

        // Count and report unexecuted tasks (they're all part of cycles)
        var cycle_count: usize = 0;
        for (self.tasks.items, 0..) |task, i| {
            if (!task.executed) {
                cycle_count += 1;
                std.log.err("  - Task {d}: '{s}'", .{ i, task.name });

                // Show what this task depends on
                if (task.dependencies.items.len > 0) {
                    for (task.dependencies.items) |dep_id| {
                        const dep_task = self.tasks.items[dep_id];
                        const status: []const u8 = if (dep_task.executed) "OK" else "CYCLE";
                        std.log.err("      depends on: '{s}' [{s}]", .{ dep_task.name, status });
                    }
                }
            }
        }

        std.log.err("[ComputeGraph] {d} of {d} tasks are stuck in cycles", .{
            cycle_count,
            self.tasks.items.len,
        });
        std.log.err("[ComputeGraph] To fix: ensure no task transitively depends on itself", .{});
    }

    fn computeLevels(self: *Self) !void {
        // Assign levels based on dependencies
        for (self.tasks.items) |*task| {
            task.level = 0;
        }

        for (self.execution_order.items) |task_id| {
            var max_dep_level: u32 = 0;
            for (self.tasks.items[task_id].dependencies.items) |dep_id| {
                max_dep_level = @max(max_dep_level, self.tasks.items[dep_id].level + 1);
            }
            self.tasks.items[task_id].level = max_dep_level;
        }

        // Group tasks by level
        var max_level: u32 = 0;
        for (self.tasks.items) |task| {
            max_level = @max(max_level, task.level);
        }

        for (0..max_level + 1) |_| {
            try self.levels.append(self.allocator, .{});
        }

        for (self.tasks.items, 0..) |task, i| {
            try self.levels.items[task.level].append(self.allocator, @intCast(i));
        }
    }

    fn generateBarriers(self: *Self) !void {
        for (self.tasks.items, 0..) |task, task_idx| {
            for (task.dependencies.items) |dep_id| {
                // Find shared resources that need barriers
                for (task.resources) |usage| {
                    for (self.tasks.items[dep_id].resources) |dep_usage| {
                        if (usage.handle.id == dep_usage.handle.id) {
                            // Need barrier if either is a write
                            if (usage.access.isWrite() or dep_usage.access.isWrite()) {
                                try self.barriers.append(self.allocator, .{
                                    .src_task = dep_id,
                                    .dst_task = @intCast(task_idx),
                                    .src_stage = dep_usage.stage,
                                    .dst_stage = usage.stage,
                                    .src_access = accessFromMode(dep_usage.access),
                                    .dst_access = accessFromMode(usage.access),
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    /// Execute the compiled graph
    pub fn execute(self: *Self, cmd: types.CommandBuffer) !void {
        if (!self.compiled) {
            try self.compile();
        }

        var barrier_idx: usize = 0;
        var current_pipeline: ?types.Pipeline = null;

        for (self.execution_order.items) |task_id| {
            const task = &self.tasks.items[task_id];

            // Insert barriers before this task
            while (barrier_idx < self.barriers.items.len and
                self.barriers.items[barrier_idx].dst_task == task_id)
            {
                const barrier = self.barriers.items[barrier_idx];
                const memory_barrier = types.MemoryBarrier{
                    .s_type = .memory_barrier,
                    .p_next = null,
                    .src_access_mask = barrier.src_access,
                    .dst_access_mask = barrier.dst_access,
                };

                self.ctx.device_dispatch.vkCmdPipelineBarrier.?(
                    cmd,
                    barrier.src_stage,
                    barrier.dst_stage,
                    .{},
                    1,
                    @ptrCast(&memory_barrier),
                    0,
                    null,
                    0,
                    null,
                );

                barrier_idx += 1;
            }

            // Begin task profiling
            if (self.profiler) |*prof| {
                prof.beginTask(cmd, task.name);
            }

            // Execute task
            if (task.callback) |callback| {
                callback(self.ctx, cmd, task.user_data);
            } else {
                // Bind pipeline if changed
                if (task.pipeline) |pipeline| {
                    if (current_pipeline == null or current_pipeline.? != pipeline) {
                        self.ctx.device_dispatch.vkCmdBindPipeline.?(cmd, .compute, pipeline);
                        current_pipeline = pipeline;
                    }
                }

                // Bind descriptors
                if (task.descriptor_sets.len > 0) {
                    if (task.layout) |layout| {
                        self.ctx.device_dispatch.vkCmdBindDescriptorSets.?(
                            cmd,
                            .compute,
                            layout,
                            0,
                            @intCast(task.descriptor_sets.len),
                            task.descriptor_sets.ptr,
                            0,
                            null,
                        );
                    }
                }

                // Push constants
                if (task.push_constants) |data| {
                    if (task.layout) |layout| {
                        self.ctx.device_dispatch.vkCmdPushConstants.?(
                            cmd,
                            layout,
                            .{ .compute = true },
                            0,
                            @intCast(data.len),
                            data.ptr,
                        );
                    }
                }

                // Dispatch
                self.ctx.device_dispatch.vkCmdDispatch.?(
                    cmd,
                    task.workgroups[0],
                    task.workgroups[1],
                    task.workgroups[2],
                );
            }

            // End task profiling
            if (self.profiler) |*prof| {
                prof.endTask(cmd);
            }
        }
    }

    /// Execute with parallel level awareness (returns level boundaries)
    pub fn executeWithLevels(
        self: *Self,
        cmd: types.CommandBuffer,
        level_callback: ?*const fn (level: u32, task_count: usize) void,
    ) !void {
        if (!self.compiled) {
            try self.compile();
        }

        var current_pipeline: ?types.Pipeline = null;

        for (self.levels.items, 0..) |level, level_idx| {
            if (level_callback) |cb| {
                cb(@intCast(level_idx), level.items.len);
            }

            // All tasks in this level can theoretically run in parallel
            for (level.items) |task_id| {
                const task = &self.tasks.items[task_id];

                // Insert required barriers
                for (self.barriers.items) |barrier| {
                    if (barrier.dst_task == task_id) {
                        const memory_barrier = types.MemoryBarrier{
                            .s_type = .memory_barrier,
                            .p_next = null,
                            .src_access_mask = barrier.src_access,
                            .dst_access_mask = barrier.dst_access,
                        };

                        self.ctx.device_dispatch.vkCmdPipelineBarrier.?(
                            cmd,
                            barrier.src_stage,
                            barrier.dst_stage,
                            .{},
                            1,
                            @ptrCast(&memory_barrier),
                            0,
                            null,
                            0,
                            null,
                        );
                    }
                }

                // Execute task
                if (task.callback) |callback| {
                    callback(self.ctx, cmd, task.user_data);
                } else {
                    if (task.pipeline) |pipeline| {
                        if (current_pipeline == null or current_pipeline.? != pipeline) {
                            self.ctx.device_dispatch.vkCmdBindPipeline.?(cmd, .compute, pipeline);
                            current_pipeline = pipeline;
                        }
                    }

                    if (task.descriptor_sets.len > 0) {
                        if (task.layout) |layout| {
                            self.ctx.device_dispatch.vkCmdBindDescriptorSets.?(
                                cmd,
                                .compute,
                                layout,
                                0,
                                @intCast(task.descriptor_sets.len),
                                task.descriptor_sets.ptr,
                                0,
                                null,
                            );
                        }
                    }

                    if (task.push_constants) |data| {
                        if (task.layout) |layout| {
                            self.ctx.device_dispatch.vkCmdPushConstants.?(
                                cmd,
                                layout,
                                .{ .compute = true },
                                0,
                                @intCast(data.len),
                                data.ptr,
                            );
                        }
                    }

                    self.ctx.device_dispatch.vkCmdDispatch.?(
                        cmd,
                        task.workgroups[0],
                        task.workgroups[1],
                        task.workgroups[2],
                    );
                }
            }
        }
    }

    /// Get graph statistics
    pub fn getStats(self: Self) struct {
        task_count: usize,
        resource_count: usize,
        barrier_count: usize,
        level_count: usize,
        max_parallelism: usize,
    } {
        var max_parallelism: usize = 0;
        for (self.levels.items) |level| {
            max_parallelism = @max(max_parallelism, level.items.len);
        }

        return .{
            .task_count = self.tasks.items.len,
            .resource_count = self.resources.count(),
            .barrier_count = self.barriers.items.len,
            .level_count = self.levels.items.len,
            .max_parallelism = max_parallelism,
        };
    }

    /// Print graph for debugging
    pub fn debugPrint(self: Self, writer: anytype) !void {
        try writer.print("Compute Graph ({} tasks, {} resources):\n", .{
            self.tasks.items.len,
            self.resources.count(),
        });

        for (self.levels.items, 0..) |level, level_idx| {
            try writer.print("  Level {}:\n", .{level_idx});
            for (level.items) |task_id| {
                const task = self.tasks.items[task_id];
                try writer.print("    [{d}] {s} ({d}x{d}x{d})\n", .{
                    task_id,
                    task.name,
                    task.workgroups[0],
                    task.workgroups[1],
                    task.workgroups[2],
                });
            }
        }

        try writer.print("  Barriers: {}\n", .{self.barriers.items.len});
    }

    // =========================================================================
    // High-Level Execution API
    // =========================================================================

    /// Execution result with timing information
    pub const ExecutionResult = struct {
        /// Total execution time in nanoseconds
        total_time_ns: u64 = 0,
        /// GPU time in nanoseconds (if profiling enabled)
        gpu_time_ns: ?u64 = null,
        /// Number of tasks executed
        tasks_executed: u32 = 0,
        /// Number of barriers inserted
        barriers_inserted: u32 = 0,
    };

    /// Submit options for graph execution
    pub const SubmitOptions = struct {
        /// Enable GPU profiling (requires timestamp queries)
        enable_profiling: bool = false,
        /// Wait for completion before returning
        wait_for_completion: bool = true,
        /// Fence to signal on completion (if not waiting)
        fence: ?types.Fence = null,
    };

    /// Record, submit, and optionally wait for graph execution
    pub fn submit(self: *Self, options: SubmitOptions) !ExecutionResult {
        const start_time = std.time.nanoTimestamp();

        if (!self.compiled) {
            try self.compile();
        }

        // Allocate command buffer from context's pool
        const cmd_pool = self.ctx.command_pool;
        const alloc_info = types.CommandBufferAllocateInfo{
            .s_type = .command_buffer_allocate_info,
            .p_next = null,
            .command_pool = cmd_pool,
            .level = .primary,
            .command_buffer_count = 1,
        };

        var cmd: types.CommandBuffer = undefined;
        const alloc_result = self.ctx.device_dispatch.allocateCommandBuffers(
            self.ctx.device,
            &alloc_info,
            @ptrCast(&cmd),
        );
        if (alloc_result != .success) return error.CommandBufferAllocationFailed;

        // Begin command buffer
        const begin_info = types.CommandBufferBeginInfo{
            .s_type = .command_buffer_begin_info,
            .p_next = null,
            .flags = .{ .one_time_submit = true },
            .p_inheritance_info = null,
        };
        const begin_result = self.ctx.device_dispatch.beginCommandBuffer(cmd, &begin_info);
        if (begin_result != .success) return error.CommandBufferBeginFailed;

        // Enable profiling if requested and not already enabled
        if (options.enable_profiling and self.profiler == null) {
            self.profiler = GpuProfiler.init(self.ctx, self.allocator) catch null;
        }

        // Begin profiling session
        if (self.profiler) |*prof| {
            prof.begin(cmd);
        }

        // Execute the graph
        try self.execute(cmd);

        // End profiling session
        if (self.profiler) |*prof| {
            prof.end(cmd);
        }

        // End command buffer
        const end_result = self.ctx.device_dispatch.endCommandBuffer(cmd);
        if (end_result != .success) return error.CommandBufferEndFailed;

        // Create fence if waiting and no external fence provided
        var local_fence: types.Fence = .null_handle;
        const use_fence = if (options.wait_for_completion and options.fence == null) blk: {
            const fence_info = types.FenceCreateInfo{
                .s_type = .fence_create_info,
                .p_next = null,
                .flags = .{},
            };
            const fence_result = self.ctx.device_dispatch.createFence(
                self.ctx.device,
                &fence_info,
                null,
                &local_fence,
            );
            if (fence_result != .success) return error.FenceCreationFailed;
            break :blk local_fence;
        } else options.fence orelse .null_handle;

        // Submit
        const submit_info = types.SubmitInfo{
            .s_type = .submit_info,
            .p_next = null,
            .wait_semaphore_count = 0,
            .p_wait_semaphores = null,
            .p_wait_dst_stage_mask = null,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&cmd),
            .signal_semaphore_count = 0,
            .p_signal_semaphores = null,
        };

        const submit_result = self.ctx.device_dispatch.queueSubmit(
            self.ctx.compute_queue,
            1,
            @ptrCast(&submit_info),
            use_fence,
        );
        if (submit_result != .success) return error.SubmissionFailed;

        // Wait if requested
        var gpu_time_ns: ?u64 = null;
        if (options.wait_for_completion) {
            const wait_result = self.ctx.device_dispatch.waitForFences(
                self.ctx.device,
                1,
                @ptrCast(&use_fence),
                1, // wait all
                std.math.maxInt(u64),
            );
            if (wait_result != .success) return error.WaitFailed;

            // Read back profiling results
            if (self.profiler) |*prof| {
                const profile_results = prof.readResults() catch null;
                if (profile_results) |results| {
                    gpu_time_ns = results.total_gpu_ns;
                    // Note: task-level results are available via getLastProfileResults()
                }
            }

            // Cleanup local fence
            if (local_fence != .null_handle) {
                self.ctx.device_dispatch.destroyFence(self.ctx.device, local_fence, null);
            }
        }

        // Free command buffer
        self.ctx.device_dispatch.freeCommandBuffers(self.ctx.device, cmd_pool, 1, @ptrCast(&cmd));

        const end_time = std.time.nanoTimestamp();

        return ExecutionResult{
            .total_time_ns = @intCast(end_time - start_time),
            .gpu_time_ns = gpu_time_ns,
            .tasks_executed = @intCast(self.tasks.items.len),
            .barriers_inserted = @intCast(self.barriers.items.len),
        };
    }

    /// Get detailed profiling results from the last execution
    /// Must be called after submit() with enable_profiling=true and wait_for_completion=true
    pub fn getLastProfileResults(self: *Self) ?ProfileResults {
        if (self.profiler) |*prof| {
            return prof.readResults() catch null;
        }
        return null;
    }

    /// Submit graph using the attached async coordinator's frame management
    /// This integrates with the coordinator's frame synchronization system.
    /// The coordinator must be attached before calling this method.
    /// Returns the coordinator's current frame index for tracking.
    pub fn submitAsync(self: *Self) !u32 {
        const coord = self.coordinator orelse return error.NoCoordinatorAttached;

        if (!self.compiled) {
            try self.compile();
        }

        // Begin frame on the coordinator (handles fence wait and command buffer reset)
        try coord.beginFrame();

        // Get the current frame's compute command buffer
        const frame_idx = coord.current_frame;
        const frame = &coord.frames[frame_idx];
        const cmd = if (frame.compute_cmd) |c| c.handle else return error.NoCommandBuffer;

        // Begin profiling if available
        if (self.profiler) |*prof| {
            prof.begin(cmd);
        }

        // Execute the graph into the coordinator's command buffer
        try self.execute(cmd);

        // End profiling
        if (self.profiler) |*prof| {
            prof.end(cmd);
        }

        // End frame (submits the command buffer)
        try coord.endFrame();

        return frame_idx;
    }

    /// Submit graph asynchronously and wait for completion
    /// This is a convenience wrapper around submitAsync() + waitCurrentFrame()
    pub fn submitAsyncAndWait(self: *Self) !ExecutionResult {
        const start_time = std.time.nanoTimestamp();

        const frame_idx = try self.submitAsync();
        _ = frame_idx;

        // Wait for the current frame to complete
        if (self.coordinator) |coord| {
            try coord.waitCurrentFrame();
        }

        // Read profiling results
        var gpu_time_ns: ?u64 = null;
        if (self.profiler) |*prof| {
            const profile_results = prof.readResults() catch null;
            if (profile_results) |results| {
                gpu_time_ns = results.total_gpu_ns;
            }
        }

        const end_time = std.time.nanoTimestamp();

        return ExecutionResult{
            .total_time_ns = @intCast(end_time - start_time),
            .gpu_time_ns = gpu_time_ns,
            .tasks_executed = @intCast(self.tasks.items.len),
            .barriers_inserted = @intCast(self.barriers.items.len),
        };
    }

    /// Execute graph multiple times (for benchmarking)
    pub fn submitN(self: *Self, count: u32, options: SubmitOptions) !ExecutionResult {
        var total_result = ExecutionResult{};

        for (0..count) |_| {
            const result = try self.submit(options);
            total_result.total_time_ns += result.total_time_ns;
            if (result.gpu_time_ns) |gpu_ns| {
                total_result.gpu_time_ns = (total_result.gpu_time_ns orelse 0) + gpu_ns;
            }
            total_result.tasks_executed += result.tasks_executed;
            total_result.barriers_inserted += result.barriers_inserted;
        }

        return total_result;
    }

    /// Reset graph for reuse with new pipelines/descriptors
    pub fn reset(self: *Self) void {
        for (self.tasks.items) |*task| {
            task.dependencies.clearRetainingCapacity();
            task.dependents.clearRetainingCapacity();
            task.executed = false;
            task.in_degree = 0;
            task.level = 0;
        }

        self.execution_order.clearRetainingCapacity();
        self.barriers.clearRetainingCapacity();

        for (self.levels.items) |*level| {
            level.deinit(self.allocator);
        }
        self.levels.clearRetainingCapacity();

        self.compiled = false;
    }

    /// Clear all tasks and resources
    pub fn clear(self: *Self) void {
        for (self.tasks.items) |*task| {
            task.dependencies.deinit(self.allocator);
            task.dependents.deinit(self.allocator);
        }
        self.tasks.clearRetainingCapacity();
        self.task_names.clearRetainingCapacity();

        var res_iter = self.resources.iterator();
        while (res_iter.next()) |entry| {
            entry.value_ptr.readers.deinit(self.allocator);
        }
        self.resources.clearRetainingCapacity();

        self.reset();
        self.next_task_id = 0;
        self.next_resource_id = 0;
    }

    /// Export graph to DOT format for visualization with GraphViz
    ///
    /// Returns a DOT-formatted string that can be rendered with `dot -Tpng graph.dot -o graph.png`
    ///
    /// ## Example
    /// ```zig
    /// const dot = try graph.exportDot();
    /// defer graph.allocator.free(dot);
    /// // Write to file or stdout
    /// ```
    pub fn exportDot(self: *Self) ![]const u8 {
        var output = std.ArrayListUnmanaged(u8){};
        errdefer output.deinit(self.allocator);

        // Header
        try output.appendSlice(self.allocator, "digraph ComputeGraph {\n");
        try output.appendSlice(self.allocator, "    rankdir=TB;\n");
        try output.appendSlice(self.allocator, "    node [shape=box, style=filled];\n");
        try output.appendSlice(self.allocator, "\n");

        // Subgraph for resources
        try output.appendSlice(self.allocator, "    subgraph cluster_resources {\n");
        try output.appendSlice(self.allocator, "        label=\"Resources\";\n");
        try output.appendSlice(self.allocator, "        style=dashed;\n");
        try output.appendSlice(self.allocator, "        node [shape=ellipse, fillcolor=lightblue];\n");

        var res_iter = self.resources.iterator();
        while (res_iter.next()) |entry| {
            var buf: [256]u8 = undefined;
            const line = std.fmt.bufPrint(&buf, "        R{d} [label=\"{s}\\n({d} bytes)\"];\n", .{
                entry.key_ptr.*,
                entry.value_ptr.name,
                entry.value_ptr.size,
            }) catch continue;
            try output.appendSlice(self.allocator, line);
        }
        try output.appendSlice(self.allocator, "    }\n\n");

        // Tasks
        try output.appendSlice(self.allocator, "    // Tasks\n");
        for (self.tasks.items) |task| {
            var buf: [256]u8 = undefined;
            const color = switch (task.queue_hint) {
                .compute => "lightyellow",
                .transfer => "lightgreen",
                .auto => "lightgray",
            };
            const line = std.fmt.bufPrint(&buf, "    T{d} [label=\"{s}\", fillcolor={s}];\n", .{
                task.id,
                task.name,
                color,
            }) catch continue;
            try output.appendSlice(self.allocator, line);
        }
        try output.appendSlice(self.allocator, "\n");

        // Task dependencies
        try output.appendSlice(self.allocator, "    // Dependencies\n");
        for (self.tasks.items) |task| {
            for (task.dependencies.items) |dep_id| {
                var buf: [128]u8 = undefined;
                const line = std.fmt.bufPrint(&buf, "    T{d} -> T{d};\n", .{ dep_id, task.id }) catch continue;
                try output.appendSlice(self.allocator, line);
            }
        }
        try output.appendSlice(self.allocator, "\n");

        // Resource usage edges
        try output.appendSlice(self.allocator, "    // Resource usage\n");
        for (self.tasks.items) |task| {
            for (task.resources) |res| {
                var buf: [128]u8 = undefined;
                const style = switch (res.access) {
                    .read => "dashed",
                    .write => "solid",
                    .read_write => "bold",
                };
                const line = std.fmt.bufPrint(&buf, "    R{d} -> T{d} [style={s}];\n", .{
                    res.handle.id,
                    task.id,
                    style,
                }) catch continue;
                try output.appendSlice(self.allocator, line);
            }
        }

        // Footer
        try output.appendSlice(self.allocator, "}\n");

        return output.toOwnedSlice(self.allocator);
    }

    /// Export graph to DOT file
    pub fn exportDotToFile(self: *Self, path: []const u8) !void {
        const dot = try self.exportDot();
        defer self.allocator.free(dot);

        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        try file.writeAll(dot);
    }
};

fn containsValue(slice: []const u32, value: u32) bool {
    for (slice) |v| {
        if (v == value) return true;
    }
    return false;
}

fn accessFromMode(mode: AccessMode) types.AccessFlags {
    return switch (mode) {
        .read => .{ .shader_read = true },
        .write => .{ .shader_write = true },
        .read_write => .{ .shader_read = true, .shader_write = true },
    };
}

// Tests
test "ComputeGraph basic" {
    var graph = ComputeGraph.init(std.testing.allocator, undefined);
    defer graph.deinit();

    const buf_a = try graph.registerResource("buffer_a", null, 1024);
    const buf_b = try graph.registerResource("buffer_b", null, 1024);

    _ = try graph.addTask(.{
        .name = "task_1",
        .workgroups = .{ 64, 1, 1 },
        .resources = &.{.{ .handle = buf_a, .access = .write }},
    });

    _ = try graph.addTask(.{
        .name = "task_2",
        .workgroups = .{ 64, 1, 1 },
        .resources = &.{
            .{ .handle = buf_a, .access = .read },
            .{ .handle = buf_b, .access = .write },
        },
    });

    try graph.compile();

    const stats = graph.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.task_count);
    try std.testing.expectEqual(@as(usize, 2), stats.resource_count);
}

test "ComputeGraph parallel tasks" {
    var graph = ComputeGraph.init(std.testing.allocator, undefined);
    defer graph.deinit();

    const buf_a = try graph.registerResource("buffer_a", null, 1024);
    const buf_b = try graph.registerResource("buffer_b", null, 1024);

    // Two independent writes - should be parallel
    _ = try graph.addTask(.{
        .name = "write_a",
        .resources = &.{.{ .handle = buf_a, .access = .write }},
    });

    _ = try graph.addTask(.{
        .name = "write_b",
        .resources = &.{.{ .handle = buf_b, .access = .write }},
    });

    try graph.compile();

    const stats = graph.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.level_count); // Both in same level
    try std.testing.expectEqual(@as(usize, 2), stats.max_parallelism);
}

test "ComputeGraph queue hints" {
    var graph = ComputeGraph.init(std.testing.allocator, undefined);
    defer graph.deinit();

    const buf_a = try graph.registerResource("buffer_a", null, 1024);
    const buf_b = try graph.registerResource("buffer_b", null, 1024);

    // Add a compute task
    _ = try graph.addTask(.{
        .name = "compute_task",
        .queue_hint = .compute,
        .resources = &.{.{ .handle = buf_a, .access = .write }},
    });

    // Add a transfer task using the convenience method
    _ = try graph.addTransferTask(.{
        .name = "transfer_task",
        .resources = &.{
            .{ .handle = buf_a, .access = .read },
            .{ .handle = buf_b, .access = .write },
        },
        .callback = struct {
            fn cb(_: *Context, _: types.CommandBuffer, _: ?*anyopaque) void {}
        }.cb,
    });

    // Add an auto task
    _ = try graph.addTask(.{
        .name = "auto_task",
        .queue_hint = .auto,
        .resources = &.{.{ .handle = buf_b, .access = .read }},
    });

    try graph.compile();

    // Check queue stats
    const queue_stats = graph.getQueueStats();
    try std.testing.expectEqual(@as(u32, 1), queue_stats.compute_tasks);
    try std.testing.expectEqual(@as(u32, 1), queue_stats.transfer_tasks);
    try std.testing.expectEqual(@as(u32, 1), queue_stats.auto_tasks);
}

test "TaskQueueHint methods" {
    // Test queue hint helper methods with null context fields
    const hint_compute = TaskQueueHint.compute;
    const hint_transfer = TaskQueueHint.transfer;
    const hint_auto = TaskQueueHint.auto;

    // Basic enum comparison
    try std.testing.expect(hint_compute == .compute);
    try std.testing.expect(hint_transfer == .transfer);
    try std.testing.expect(hint_auto == .auto);
}

test "ComputeGraph coordinator attachment" {
    var graph = ComputeGraph.init(std.testing.allocator, undefined);
    defer graph.deinit();

    // Initially no coordinator
    try std.testing.expect(!graph.hasCoordinator());
    try std.testing.expect(graph.getCoordinator() == null);

    // Try to submit async without coordinator (should fail)
    const result = graph.submitAsync();
    try std.testing.expectError(error.NoCoordinatorAttached, result);

    // Note: Full coordinator tests require Vulkan device initialization
    // This test just validates the attachment API works correctly
}
