//! Vulkan Compute Benchmarks
//!
//! Measures performance of various Vulkan compute operations.
//! Results are output in both human-readable and JSON formats.
//!
//! ## Regression Tracking
//!
//! Use `--check-regression` to compare against baseline and detect regressions.
//! Use `--save-baseline` to save current results as the new baseline.
//! Use `--baseline <path>` to specify a custom baseline file.

const std = @import("std");
const vulkan = @import("vulkan");
const regression = @import("regression.zig");

const BenchmarkResult = struct {
    name: []const u8,
    iterations: u64,
    total_ns: u64,
    min_ns: u64,
    max_ns: u64,
    avg_ns: u64,
    throughput_mb_s: ?f64,

    /// Convert to regression tracker format
    pub fn toRegressionResult(self: BenchmarkResult) regression.RegressionTracker.BenchResult {
        return .{
            .name = self.name,
            .avg_ns = self.avg_ns,
            .min_ns = self.min_ns,
            .max_ns = self.max_ns,
            .iterations = self.iterations,
        };
    }
};

const BenchmarkSuite = struct {
    results: std.ArrayListUnmanaged(BenchmarkResult),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) BenchmarkSuite {
        return .{
            .results = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BenchmarkSuite) void {
        self.results.deinit(self.allocator);
    }

    pub fn add(self: *BenchmarkSuite, result: BenchmarkResult) !void {
        try self.results.append(self.allocator, result);
    }

    pub fn printTable(self: BenchmarkSuite) void {
        std.debug.print("\n", .{});
        std.debug.print("========================================================================\n", .{});
        std.debug.print("                         BENCHMARK RESULTS                              \n", .{});
        std.debug.print("========================================================================\n", .{});
        std.debug.print("| {s:<26} | {s:>9} | {s:>9} | {s:>9} | {s:>11} |\n", .{ "Name", "Avg (us)", "Min (us)", "Max (us)", "Throughput" });
        std.debug.print("|----------------------------|-----------|-----------|-----------|-------------|\n", .{});

        for (self.results.items) |r| {
            const avg_us = @as(f64, @floatFromInt(r.avg_ns)) / 1000.0;
            const min_us = @as(f64, @floatFromInt(r.min_ns)) / 1000.0;
            const max_us = @as(f64, @floatFromInt(r.max_ns)) / 1000.0;

            if (r.throughput_mb_s) |tp| {
                std.debug.print("| {s:<26} | {d:>9.2} | {d:>9.2} | {d:>9.2} | {d:>7.1} MB/s |\n", .{
                    r.name,
                    avg_us,
                    min_us,
                    max_us,
                    tp,
                });
            } else {
                std.debug.print("| {s:<26} | {d:>9.2} | {d:>9.2} | {d:>9.2} |      N/A    |\n", .{
                    r.name,
                    avg_us,
                    min_us,
                    max_us,
                });
            }
        }

        std.debug.print("========================================================================\n", .{});
    }

    pub fn printJson(self: BenchmarkSuite, writer: anytype) !void {
        try writer.writeAll("{\n  \"benchmarks\": [\n");

        for (self.results.items, 0..) |r, i| {
            try writer.print("    {{\n", .{});
            try writer.print("      \"name\": \"{s}\",\n", .{r.name});
            try writer.print("      \"iterations\": {},\n", .{r.iterations});
            try writer.print("      \"total_ns\": {},\n", .{r.total_ns});
            try writer.print("      \"min_ns\": {},\n", .{r.min_ns});
            try writer.print("      \"max_ns\": {},\n", .{r.max_ns});
            try writer.print("      \"avg_ns\": {},\n", .{r.avg_ns});
            if (r.throughput_mb_s) |tp| {
                try writer.print("      \"throughput_mb_s\": {d:.2}\n", .{tp});
            } else {
                try writer.print("      \"throughput_mb_s\": null\n", .{});
            }
            try writer.print("    }}{s}\n", .{if (i < self.results.items.len - 1) "," else ""});
        }

        try writer.writeAll("  ]\n}\n");
    }
};

fn runBenchmark(
    name: []const u8,
    iterations: u64,
    size_bytes: ?u64,
    func: *const fn () anyerror!void,
) BenchmarkResult {
    var min_ns: u64 = std.math.maxInt(u64);
    var max_ns: u64 = 0;
    var total_ns: u64 = 0;

    // Warmup
    for (0..3) |_| {
        func() catch {};
    }

    // Actual benchmark
    for (0..iterations) |_| {
        const start = std.time.nanoTimestamp();
        func() catch {};
        const end = std.time.nanoTimestamp();

        const elapsed: u64 = @intCast(end - start);
        total_ns += elapsed;
        min_ns = @min(min_ns, elapsed);
        max_ns = @max(max_ns, elapsed);
    }

    const avg_ns = total_ns / iterations;

    const throughput: ?f64 = if (size_bytes) |sz| blk: {
        const bytes_per_sec = @as(f64, @floatFromInt(sz)) / (@as(f64, @floatFromInt(avg_ns)) / 1_000_000_000.0);
        break :blk bytes_per_sec / (1024.0 * 1024.0);
    } else null;

    return .{
        .name = name,
        .iterations = iterations,
        .total_ns = total_ns,
        .min_ns = min_ns,
        .max_ns = max_ns,
        .avg_ns = avg_ns,
        .throughput_mb_s = throughput,
    };
}

// Global state for benchmarks (needed due to function pointer constraints)
var g_ctx: ?*vulkan.Context = null;
var g_allocator: std.mem.Allocator = undefined;

fn benchContextCreation() !void {
    var ctx = try vulkan.Context.init(g_allocator, .{});
    ctx.deinit();
}

fn benchBufferCreate1KB() !void {
    var buf = try vulkan.Buffer.init(g_ctx.?, 1024, .staging);
    buf.deinit(g_ctx.?);
}

fn benchBufferCreate1MB() !void {
    var buf = try vulkan.Buffer.init(g_ctx.?, 1024 * 1024, .staging);
    buf.deinit(g_ctx.?);
}

fn benchBufferCreate16MB() !void {
    var buf = try vulkan.Buffer.init(g_ctx.?, 16 * 1024 * 1024, .staging);
    buf.deinit(g_ctx.?);
}

fn benchEmptySubmit() !void {
    var cmd = try vulkan.CommandBuffer.init(g_ctx.?);
    defer cmd.deinit();

    try cmd.begin();
    try cmd.end();

    try vulkan.Submission.submitAndWait(g_ctx.?, cmd);
}

fn benchFenceWait() !void {
    var fence = try vulkan.Fence.init(g_ctx.?, false);
    defer fence.deinit();

    var cmd = try vulkan.CommandBuffer.init(g_ctx.?);
    defer cmd.deinit();

    try cmd.begin();
    try cmd.end();

    try vulkan.Submission.submit(g_ctx.?, cmd, null, null, fence);
    try fence.waitForever();
}

// Global buffers for transfer benchmarks
var g_staging_1mb: ?vulkan.Buffer = null;
var g_device_1mb: ?vulkan.Buffer = null;
var g_readback_1mb: ?vulkan.Buffer = null;
var g_staging_16mb: ?vulkan.Buffer = null;
var g_device_16mb: ?vulkan.Buffer = null;
var g_readback_16mb: ?vulkan.Buffer = null;

fn benchH2DTransfer1MB() !void {
    var cmd = try vulkan.CommandBuffer.init(g_ctx.?);
    defer cmd.deinit();

    try cmd.begin();
    vulkan.Transfer.copy(g_ctx.?.device_dispatch, cmd.getHandle(), g_staging_1mb.?, g_device_1mb.?, null);
    try cmd.end();

    try vulkan.Submission.submitAndWait(g_ctx.?, cmd);
}

fn benchD2HTransfer1MB() !void {
    var cmd = try vulkan.CommandBuffer.init(g_ctx.?);
    defer cmd.deinit();

    try cmd.begin();
    vulkan.Transfer.copy(g_ctx.?.device_dispatch, cmd.getHandle(), g_device_1mb.?, g_readback_1mb.?, null);
    try cmd.end();

    try vulkan.Submission.submitAndWait(g_ctx.?, cmd);
}

fn benchH2DTransfer16MB() !void {
    var cmd = try vulkan.CommandBuffer.init(g_ctx.?);
    defer cmd.deinit();

    try cmd.begin();
    vulkan.Transfer.copy(g_ctx.?.device_dispatch, cmd.getHandle(), g_staging_16mb.?, g_device_16mb.?, null);
    try cmd.end();

    try vulkan.Submission.submitAndWait(g_ctx.?, cmd);
}

fn benchD2HTransfer16MB() !void {
    var cmd = try vulkan.CommandBuffer.init(g_ctx.?);
    defer cmd.deinit();

    try cmd.begin();
    vulkan.Transfer.copy(g_ctx.?.device_dispatch, cmd.getHandle(), g_device_16mb.?, g_readback_16mb.?, null);
    try cmd.end();

    try vulkan.Submission.submitAndWait(g_ctx.?, cmd);
}

fn benchDescriptorPoolAlloc() !void {
    const bindings = [_]vulkan.DescriptorBinding{
        .{ .binding = 0, .descriptor_type = .storage_buffer },
        .{ .binding = 1, .descriptor_type = .storage_buffer },
    };

    var layout = try vulkan.DescriptorSetLayout.init(g_ctx.?, &bindings);
    defer layout.deinit();

    var pool = try vulkan.DescriptorPool.initSimple(g_ctx.?, 16, 32);
    defer pool.deinit();

    // Allocate and free multiple sets
    var sets: [8]vulkan.vk.DescriptorSet = undefined;
    for (&sets) |*set| {
        set.* = try pool.allocate(layout);
    }
    for (sets) |set| {
        pool.free(set);
    }
}

fn benchSpirvValidation() !void {
    // Larger synthetic SPIR-V for validation benchmark
    const spirv = [_]u32{
        0x07230203, // Magic
        0x00010300, // Version 1.3
        0x0001_0001, // Generator
        0x00000100, // Bound = 256
        0x00000000, // Schema
    } ++ [_]u32{0} ** 100; // Pad with NOP-like words

    _ = vulkan.validateSpirv(&spirv);
}

// Mock allocator benchmarks (always work, no GPU needed)
var g_mock_ctx: ?vulkan.MockContext = null;

fn benchMockAllocate1KB() !void {
    if (g_mock_ctx) |*mock| {
        const handle = try mock.allocateMemory(1024, 0);
        mock.freeMemory(handle);
    }
}

fn benchMockAllocate1MB() !void {
    if (g_mock_ctx) |*mock| {
        const handle = try mock.allocateMemory(1024 * 1024, 0);
        mock.freeMemory(handle);
    }
}

fn benchMockBuffer() !void {
    if (g_mock_ctx) |*mock| {
        _ = try mock.createBuffer(4096);
    }
}

fn benchMockErrorInjection() !void {
    if (g_mock_ctx) |*mock| {
        try mock.injectError(.allocateMemory, vulkan.Error.OutOfDeviceMemory);
        _ = mock.allocateMemory(1024, 0) catch {};
    }
}

// Compute graph benchmarks
var g_graph_ctx: ?*vulkan.Context = null;
var g_graph: ?vulkan.ComputeGraph = null;

fn setupGraphBenchmark() !void {
    if (g_graph_ctx) |ctx| {
        if (g_graph) |*graph| {
            graph.clear();
        } else {
            g_graph = vulkan.ComputeGraph.init(g_allocator, ctx);
        }

        var graph = &(g_graph.?);

        // Create a realistic ML-like graph
        const input = try graph.registerResource("input", null, 16 * 1024 * 1024);
        const weights = try graph.registerResource("weights", null, 4 * 1024 * 1024);
        const output1 = try graph.registerResource("hidden", null, 8 * 1024 * 1024);
        const output2 = try graph.registerResource("output", null, 4 * 1024 * 1024);

        // Layer 1: input + weights -> hidden
        _ = try graph.addTask(.{
            .name = "layer1",
            .workgroups = .{ 256, 1, 1 },
            .resources = &.{
                .{ .handle = input, .access = .read },
                .{ .handle = weights, .access = .read },
                .{ .handle = output1, .access = .write },
            },
        });

        // Activation
        _ = try graph.addTask(.{
            .name = "activation",
            .workgroups = .{ 256, 1, 1 },
            .resources = &.{
                .{ .handle = output1, .access = .read_write },
            },
        });

        // Layer 2: hidden + weights -> output
        _ = try graph.addTask(.{
            .name = "layer2",
            .workgroups = .{ 128, 1, 1 },
            .resources = &.{
                .{ .handle = output1, .access = .read },
                .{ .handle = weights, .access = .read },
                .{ .handle = output2, .access = .write },
            },
        });

        // Softmax
        _ = try graph.addTask(.{
            .name = "softmax",
            .workgroups = .{ 32, 1, 1 },
            .resources = &.{
                .{ .handle = output2, .access = .read_write },
            },
        });
    }
}

fn benchGraphCompile() !void {
    try setupGraphBenchmark();
    if (g_graph) |*graph| {
        graph.reset();
        try graph.compile();
    }
}

fn benchCompilerDetection() !void {
    _ = vulkan.glsl.detectCompilers(g_allocator);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    g_allocator = allocator;

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    // Parse command-line arguments
    var json_output = false;
    var check_regression = false;
    var save_baseline = false;
    var baseline_path: []const u8 = "bench/baseline.json";
    var regression_threshold: f64 = 0.10;
    var show_help = false;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--json")) {
            json_output = true;
        } else if (std.mem.eql(u8, arg, "--check-regression")) {
            check_regression = true;
        } else if (std.mem.eql(u8, arg, "--save-baseline")) {
            save_baseline = true;
        } else if (std.mem.eql(u8, arg, "--baseline")) {
            i += 1;
            if (i < args.len) {
                baseline_path = args[i];
            }
        } else if (std.mem.eql(u8, arg, "--threshold")) {
            i += 1;
            if (i < args.len) {
                regression_threshold = std.fmt.parseFloat(f64, args[i]) catch 0.10;
            }
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            show_help = true;
        }
    }

    if (show_help) {
        std.debug.print(
            \\Vulkan Compute Benchmarks
            \\
            \\Usage: bench [OPTIONS]
            \\
            \\Options:
            \\  --json                Output results in JSON format
            \\  --check-regression    Compare results against baseline and report regressions
            \\  --save-baseline       Save current results as the new baseline
            \\  --baseline <path>     Path to baseline file (default: bench/baseline.json)
            \\  --threshold <pct>     Regression threshold as decimal (default: 0.10 = 10%)
            \\  --help, -h            Show this help message
            \\
            \\Examples:
            \\  bench                           Run benchmarks with human-readable output
            \\  bench --json                    Run benchmarks with JSON output
            \\  bench --save-baseline           Run benchmarks and save as baseline
            \\  bench --check-regression        Run benchmarks and check for regressions
            \\  bench --threshold 0.05          Use 5% regression threshold
            \\
        , .{});
        return;
    }

    var suite = BenchmarkSuite.init(allocator);
    defer suite.deinit();

    // Check Vulkan availability
    if (!vulkan.isAvailable()) {
        std.debug.print("Vulkan not available - cannot run benchmarks\n", .{});
        std.process.exit(1);
    }

    // Print device info
    var probe_result = vulkan.probe(allocator);
    defer probe_result.deinit(allocator);

    if (!probe_result.isAvailable()) {
        std.debug.print("No Vulkan devices found - cannot run benchmarks\n", .{});
        std.process.exit(1);
    }

    if (!json_output) {
        std.debug.print("Vulkan Compute Benchmarks\n", .{});
        std.debug.print("========================\n\n", .{});

        for (probe_result.devices) |dev| {
            if (dev.has_compute_queue) {
                std.debug.print("Device: {s}\n", .{dev.getName()});
                std.debug.print("Type: {s}\n", .{@tagName(dev.device_type)});
                std.debug.print("Memory: {} MB\n\n", .{dev.total_memory / (1024 * 1024)});
                break;
            }
        }
    }

    // Create persistent context for most benchmarks
    var ctx = vulkan.Context.init(allocator, .{}) catch |e| {
        std.debug.print("Failed to create context: {}\n", .{e});
        std.process.exit(1);
    };
    defer ctx.deinit();
    g_ctx = &ctx;

    // Run benchmarks
    if (!json_output) std.debug.print("Running benchmarks...\n", .{});

    // Context creation benchmark (uses separate context)
    g_ctx = null;
    try suite.add(runBenchmark("Context creation", 10, null, benchContextCreation));
    g_ctx = &ctx;

    // Buffer creation benchmarks
    try suite.add(runBenchmark("Buffer create 1KB", 100, null, benchBufferCreate1KB));
    try suite.add(runBenchmark("Buffer create 1MB", 50, null, benchBufferCreate1MB));
    try suite.add(runBenchmark("Buffer create 16MB", 20, null, benchBufferCreate16MB));

    // Submission benchmarks
    try suite.add(runBenchmark("Empty submit", 100, null, benchEmptySubmit));
    try suite.add(runBenchmark("Fence wait", 100, null, benchFenceWait));

    // Memory transfer benchmarks (create persistent buffers)
    g_staging_1mb = try vulkan.Buffer.init(&ctx, 1024 * 1024, .staging);
    defer g_staging_1mb.?.deinit(&ctx);

    g_device_1mb = try vulkan.Buffer.init(&ctx, 1024 * 1024, .storage);
    defer g_device_1mb.?.deinit(&ctx);

    g_readback_1mb = try vulkan.Buffer.init(&ctx, 1024 * 1024, .readback);
    defer g_readback_1mb.?.deinit(&ctx);

    g_staging_16mb = try vulkan.Buffer.init(&ctx, 16 * 1024 * 1024, .staging);
    defer g_staging_16mb.?.deinit(&ctx);

    g_device_16mb = try vulkan.Buffer.init(&ctx, 16 * 1024 * 1024, .storage);
    defer g_device_16mb.?.deinit(&ctx);

    g_readback_16mb = try vulkan.Buffer.init(&ctx, 16 * 1024 * 1024, .readback);
    defer g_readback_16mb.?.deinit(&ctx);

    try suite.add(runBenchmark("H2D transfer 1MB", 50, 1024 * 1024, benchH2DTransfer1MB));
    try suite.add(runBenchmark("D2H transfer 1MB", 50, 1024 * 1024, benchD2HTransfer1MB));
    try suite.add(runBenchmark("H2D transfer 16MB", 20, 16 * 1024 * 1024, benchH2DTransfer16MB));
    try suite.add(runBenchmark("D2H transfer 16MB", 20, 16 * 1024 * 1024, benchD2HTransfer16MB));

    // Descriptor pool benchmark
    try suite.add(runBenchmark("Descriptor alloc 8x", 50, null, benchDescriptorPoolAlloc));

    // SPIR-V validation benchmark (CPU-only, always works)
    try suite.add(runBenchmark("SPIR-V validation", 10000, null, benchSpirvValidation));

    // Compiler detection benchmark (CPU-only)
    try suite.add(runBenchmark("Compiler detection", 100, null, benchCompilerDetection));

    // Mock context benchmarks (CPU-only, always work)
    if (!json_output) std.debug.print("Running mock context benchmarks...\n", .{});
    g_mock_ctx = vulkan.MockContext.init(allocator, .{});
    defer if (g_mock_ctx) |*mock| mock.deinit();

    try suite.add(runBenchmark("Mock alloc 1KB", 10000, null, benchMockAllocate1KB));
    try suite.add(runBenchmark("Mock alloc 1MB", 10000, null, benchMockAllocate1MB));
    try suite.add(runBenchmark("Mock buffer create", 10000, null, benchMockBuffer));
    try suite.add(runBenchmark("Mock error inject", 5000, null, benchMockErrorInjection));

    // Compute graph benchmarks
    if (!json_output) std.debug.print("Running compute graph benchmarks...\n", .{});
    g_graph_ctx = &ctx;
    defer {
        if (g_graph) |*graph| graph.deinit();
    }

    try suite.add(runBenchmark("Graph compile (4 tasks)", 1000, null, benchGraphCompile));

    // Output results
    if (json_output) {
        // For JSON output, write to a buffer and then to stderr
        var buf: [16384]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        try suite.printJson(fbs.writer());
        std.debug.print("{s}", .{fbs.getWritten()});
    } else {
        suite.printTable();
    }

    // Regression tracking
    if (check_regression or save_baseline) {
        var tracker = regression.RegressionTracker.init(allocator, .{
            .baseline_path = baseline_path,
            .regression_threshold = regression_threshold,
            .improvement_threshold = regression_threshold,
        }) catch |e| {
            std.debug.print("Failed to initialize regression tracker: {}\n", .{e});
            std.process.exit(1);
        };
        defer tracker.deinit();

        // Convert results to regression format
        var regression_results = std.ArrayListUnmanaged(regression.RegressionTracker.BenchResult){};
        defer regression_results.deinit(allocator);

        for (suite.results.items) |r| {
            regression_results.append(allocator, r.toRegressionResult()) catch {};
        }

        if (save_baseline) {
            tracker.saveBaseline(regression_results.items) catch |e| {
                std.debug.print("Failed to save baseline: {}\n", .{e});
                std.process.exit(1);
            };
            std.debug.print("\nBaseline saved to: {s}\n", .{baseline_path});
            std.debug.print("Saved {} benchmark results.\n", .{regression_results.items.len});
        }

        if (check_regression) {
            var report = tracker.compare(regression_results.items) catch |e| {
                std.debug.print("Failed to compare against baseline: {}\n", .{e});
                std.process.exit(1);
            };
            defer report.deinit();

            if (!json_output) {
                report.printSummary();
            } else {
                // JSON regression output
                var buf: [32768]u8 = undefined;
                var fbs = std.io.fixedBufferStream(&buf);
                report.toJson(fbs.writer()) catch {};
                std.debug.print("{s}", .{fbs.getWritten()});
            }

            // Exit with error if regressions detected
            if (report.hasRegressions()) {
                std.debug.print("\nExiting with error due to detected regressions.\n", .{});
                std.process.exit(1);
            }
        }
    }
}
