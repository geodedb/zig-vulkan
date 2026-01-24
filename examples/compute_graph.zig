//! Compute Graph Example
//!
//! Demonstrates DAG-based compute task scheduling with automatic
//! barrier insertion and parallel execution detection.

const std = @import("std");
const vulkan = @import("vulkan");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Compute Graph Example\n", .{});
    std.debug.print("=====================\n\n", .{});

    // Check availability
    if (!vulkan.isAvailable()) {
        std.debug.print("Vulkan not available\n", .{});
        return;
    }

    // Create context
    var ctx = vulkan.Context.init(allocator, .{
        .app_name = "compute_graph_example",
        .enable_validation = true,
    }) catch |e| {
        std.debug.print("Failed to create context: {s}\n", .{vulkan.vk.describe(e)});
        return;
    };
    defer ctx.deinit();

    std.debug.print("Using device: {s}\n\n", .{ctx.deviceInfo().getName()});

    // Create compute graph
    var graph = vulkan.ComputeGraph.init(allocator, &ctx);
    defer graph.deinit();

    // Register resources (simulating an ML inference pipeline)
    std.debug.print("Creating ML inference graph...\n", .{});

    const input_tensor = try graph.registerResource("input_tensor", null, 4 * 1024 * 1024); // 4MB input
    const embedding = try graph.registerResource("embedding", null, 2 * 1024 * 1024); // 2MB embedding
    const attention_q = try graph.registerResource("attention_Q", null, 1024 * 1024); // 1MB Q
    const attention_k = try graph.registerResource("attention_K", null, 1024 * 1024); // 1MB K
    const attention_v = try graph.registerResource("attention_V", null, 1024 * 1024); // 1MB V
    const attention_out = try graph.registerResource("attention_output", null, 1024 * 1024);
    const mlp_hidden = try graph.registerResource("mlp_hidden", null, 4 * 1024 * 1024);
    const output = try graph.registerResource("output", null, 1024 * 1024);

    // Add tasks (each represents a compute shader dispatch)

    // 1. Embedding lookup
    _ = try graph.addTask(.{
        .name = "embedding_lookup",
        .workgroups = .{ 128, 1, 1 },
        .resources = &.{
            .{ .handle = input_tensor, .access = .read },
            .{ .handle = embedding, .access = .write },
        },
    });

    // 2. Attention Q/K/V projections (can run in parallel!)
    _ = try graph.addTask(.{
        .name = "compute_Q",
        .workgroups = .{ 64, 1, 1 },
        .resources = &.{
            .{ .handle = embedding, .access = .read },
            .{ .handle = attention_q, .access = .write },
        },
    });

    _ = try graph.addTask(.{
        .name = "compute_K",
        .workgroups = .{ 64, 1, 1 },
        .resources = &.{
            .{ .handle = embedding, .access = .read },
            .{ .handle = attention_k, .access = .write },
        },
    });

    _ = try graph.addTask(.{
        .name = "compute_V",
        .workgroups = .{ 64, 1, 1 },
        .resources = &.{
            .{ .handle = embedding, .access = .read },
            .{ .handle = attention_v, .access = .write },
        },
    });

    // 3. Attention output
    _ = try graph.addTask(.{
        .name = "attention_combine",
        .workgroups = .{ 64, 1, 1 },
        .resources = &.{
            .{ .handle = attention_q, .access = .read },
            .{ .handle = attention_k, .access = .read },
            .{ .handle = attention_v, .access = .read },
            .{ .handle = attention_out, .access = .write },
        },
    });

    // 4. MLP
    _ = try graph.addTask(.{
        .name = "mlp_expand",
        .workgroups = .{ 256, 1, 1 },
        .resources = &.{
            .{ .handle = attention_out, .access = .read },
            .{ .handle = mlp_hidden, .access = .write },
        },
    });

    _ = try graph.addTask(.{
        .name = "mlp_activation",
        .workgroups = .{ 256, 1, 1 },
        .resources = &.{
            .{ .handle = mlp_hidden, .access = .read_write },
        },
    });

    _ = try graph.addTask(.{
        .name = "mlp_project",
        .workgroups = .{ 64, 1, 1 },
        .resources = &.{
            .{ .handle = mlp_hidden, .access = .read },
            .{ .handle = output, .access = .write },
        },
    });

    // Compile the graph
    std.debug.print("Compiling graph...\n", .{});
    try graph.compile();

    // Print graph stats
    const stats = graph.getStats();
    std.debug.print("\nGraph Statistics:\n", .{});
    std.debug.print("  Tasks: {}\n", .{stats.task_count});
    std.debug.print("  Resources: {}\n", .{stats.resource_count});
    std.debug.print("  Barriers: {}\n", .{stats.barrier_count});
    std.debug.print("  Execution levels: {}\n", .{stats.level_count});
    std.debug.print("  Max parallelism: {}\n", .{stats.max_parallelism});

    // Print execution order
    std.debug.print("\nExecution Order (by level):\n", .{});
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();

    try graph.debugPrint(buf.writer());
    std.debug.print("{s}\n", .{buf.items});

    std.debug.print("Key Insight: Q/K/V projections run in parallel (level 2)\n", .{});
    std.debug.print("Automatic barrier insertion ensures correctness.\n", .{});

    std.debug.print("\nExample completed successfully.\n", .{});
}
