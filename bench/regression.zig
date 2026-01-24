//! Benchmark Regression Tracking
//!
//! Provides tools for detecting performance regressions by comparing
//! benchmark results against stored baselines.
//!
//! ## Usage
//!
//! ```zig
//! var tracker = try RegressionTracker.init(allocator, .{});
//! defer tracker.deinit();
//!
//! // Run benchmarks and collect results...
//!
//! // Check for regressions
//! const report = try tracker.compare(current_results);
//! report.printSummary();
//!
//! if (report.hasRegressions()) {
//!     // Fail CI
//!     std.process.exit(1);
//! }
//!
//! // Update baseline if desired
//! try tracker.saveBaseline(current_results);
//! ```

const std = @import("std");

/// Configuration for regression detection
pub const RegressionConfig = struct {
    /// Path to the baseline file (JSON)
    baseline_path: []const u8 = "bench/baseline.json",

    /// Threshold for regression (as percentage increase)
    /// e.g., 0.10 = 10% slower triggers regression
    regression_threshold: f64 = 0.10,

    /// Threshold for improvement (as percentage decrease)
    /// e.g., 0.10 = 10% faster triggers improvement
    improvement_threshold: f64 = 0.10,

    /// Threshold for noise (ignore changes below this percentage)
    noise_threshold: f64 = 0.03,

    /// Maximum acceptable coefficient of variation
    /// Results with higher variance are flagged as unreliable
    max_cv: f64 = 0.20,

    /// Minimum number of iterations for reliable results
    min_iterations: u64 = 10,
};

/// Result of comparing a single benchmark against baseline
pub const ComparisonResult = struct {
    name: []const u8,
    baseline_ns: u64,
    current_ns: u64,
    change_pct: f64,
    status: Status,
    notes: []const u8,

    pub const Status = enum {
        regression,
        improvement,
        stable,
        new,
        missing,
        unreliable,
    };

    pub fn printLine(self: ComparisonResult) void {
        const symbol = switch (self.status) {
            .regression => "!",
            .improvement => "+",
            .stable => " ",
            .new => "*",
            .missing => "?",
            .unreliable => "~",
        };

        const status_str = switch (self.status) {
            .regression => "REGRESSION",
            .improvement => "IMPROVED",
            .stable => "STABLE",
            .new => "NEW",
            .missing => "MISSING",
            .unreliable => "UNRELIABLE",
        };

        std.debug.print("[{s}] {s:<26} ", .{ symbol, self.name });

        if (self.status == .new) {
            std.debug.print("{d:>9.2} us (NEW)\n", .{@as(f64, @floatFromInt(self.current_ns)) / 1000.0});
        } else if (self.status == .missing) {
            std.debug.print("baseline: {d:>9.2} us (MISSING)\n", .{@as(f64, @floatFromInt(self.baseline_ns)) / 1000.0});
        } else {
            const baseline_us = @as(f64, @floatFromInt(self.baseline_ns)) / 1000.0;
            const current_us = @as(f64, @floatFromInt(self.current_ns)) / 1000.0;
            const change = self.change_pct * 100.0;
            const sign: []const u8 = if (change >= 0) "+" else "";
            std.debug.print("{d:>9.2} -> {d:>9.2} us ({s}{d:5.1}%) {s}", .{
                baseline_us,
                current_us,
                sign,
                change,
                status_str,
            });
            if (self.notes.len > 0) {
                std.debug.print(" ({s})", .{self.notes});
            }
            std.debug.print("\n", .{});
        }
    }
};

/// Full regression report
pub const RegressionReport = struct {
    comparisons: std.ArrayListUnmanaged(ComparisonResult),
    allocator: std.mem.Allocator,
    config: RegressionConfig,

    pub fn init(allocator: std.mem.Allocator, config: RegressionConfig) RegressionReport {
        return .{
            .comparisons = .{},
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *RegressionReport) void {
        self.comparisons.deinit(self.allocator);
    }

    pub fn add(self: *RegressionReport, result: ComparisonResult) !void {
        try self.comparisons.append(self.allocator, result);
    }

    pub fn hasRegressions(self: RegressionReport) bool {
        for (self.comparisons.items) |c| {
            if (c.status == .regression) return true;
        }
        return false;
    }

    pub fn countByStatus(self: RegressionReport, status: ComparisonResult.Status) usize {
        var count: usize = 0;
        for (self.comparisons.items) |c| {
            if (c.status == status) count += 1;
        }
        return count;
    }

    pub fn printSummary(self: RegressionReport) void {
        const regressions = self.countByStatus(.regression);
        const improvements = self.countByStatus(.improvement);
        const stable = self.countByStatus(.stable);
        const new = self.countByStatus(.new);
        const unreliable = self.countByStatus(.unreliable);

        std.debug.print("\n", .{});
        std.debug.print("========================================================================\n", .{});
        std.debug.print("                      REGRESSION ANALYSIS                              \n", .{});
        std.debug.print("========================================================================\n", .{});
        std.debug.print("Threshold: >{d:.0}% = regression, <-{d:.0}% = improvement\n", .{
            self.config.regression_threshold * 100.0,
            self.config.improvement_threshold * 100.0,
        });
        std.debug.print("------------------------------------------------------------------------\n", .{});

        // Print all comparisons grouped by status
        if (regressions > 0) {
            std.debug.print("\nREGRESSIONS ({}):\n", .{regressions});
            for (self.comparisons.items) |c| {
                if (c.status == .regression) c.printLine();
            }
        }

        if (improvements > 0) {
            std.debug.print("\nIMPROVEMENTS ({}):\n", .{improvements});
            for (self.comparisons.items) |c| {
                if (c.status == .improvement) c.printLine();
            }
        }

        if (new > 0) {
            std.debug.print("\nNEW BENCHMARKS ({}):\n", .{new});
            for (self.comparisons.items) |c| {
                if (c.status == .new) c.printLine();
            }
        }

        if (unreliable > 0) {
            std.debug.print("\nUNRELIABLE ({}):\n", .{unreliable});
            for (self.comparisons.items) |c| {
                if (c.status == .unreliable) c.printLine();
            }
        }

        std.debug.print("\n------------------------------------------------------------------------\n", .{});
        std.debug.print("Summary: {} regressions, {} improvements, {} stable, {} new, {} unreliable\n", .{
            regressions,
            improvements,
            stable,
            new,
            unreliable,
        });

        if (regressions > 0) {
            std.debug.print("\n*** REGRESSIONS DETECTED ***\n", .{});
        } else {
            std.debug.print("\n*** NO REGRESSIONS ***\n", .{});
        }
        std.debug.print("========================================================================\n", .{});
    }

    pub fn toJson(self: RegressionReport, writer: anytype) !void {
        try writer.writeAll("{\n");
        try writer.print("  \"threshold_regression\": {d:.2},\n", .{self.config.regression_threshold});
        try writer.print("  \"threshold_improvement\": {d:.2},\n", .{self.config.improvement_threshold});
        try writer.print("  \"has_regressions\": {},\n", .{self.hasRegressions()});
        try writer.writeAll("  \"comparisons\": [\n");

        for (self.comparisons.items, 0..) |c, i| {
            try writer.writeAll("    {\n");
            try writer.print("      \"name\": \"{s}\",\n", .{c.name});
            try writer.print("      \"baseline_ns\": {},\n", .{c.baseline_ns});
            try writer.print("      \"current_ns\": {},\n", .{c.current_ns});
            try writer.print("      \"change_pct\": {d:.4},\n", .{c.change_pct});
            try writer.print("      \"status\": \"{s}\"\n", .{@tagName(c.status)});
            try writer.print("    }}{s}\n", .{if (i < self.comparisons.items.len - 1) "," else ""});
        }

        try writer.writeAll("  ]\n}\n");
    }
};

/// Baseline entry for a single benchmark
pub const BaselineEntry = struct {
    name: []const u8,
    avg_ns: u64,
    min_ns: u64,
    max_ns: u64,
    iterations: u64,
    timestamp: i64,
};

/// Regression tracker for comparing benchmarks against baselines
pub const RegressionTracker = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    config: RegressionConfig,
    baseline: std.StringHashMapUnmanaged(BaselineEntry),
    baseline_loaded: bool,

    pub fn init(allocator: std.mem.Allocator, config: RegressionConfig) !Self {
        var self = Self{
            .allocator = allocator,
            .config = config,
            .baseline = .{},
            .baseline_loaded = false,
        };

        // Try to load existing baseline
        self.loadBaseline() catch |e| {
            if (e != error.FileNotFound) {
                std.debug.print("Warning: Failed to load baseline: {}\n", .{e});
            }
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        var it = self.baseline.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.baseline.deinit(self.allocator);
    }

    /// Load baseline from JSON file
    pub fn loadBaseline(self: *Self) !void {
        const file = try std.fs.cwd().openFile(self.config.baseline_path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(self.allocator, 10 * 1024 * 1024);
        defer self.allocator.free(content);

        try self.parseBaseline(content);
        self.baseline_loaded = true;
    }

    /// Parse baseline JSON content
    fn parseBaseline(self: *Self, content: []const u8) !void {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, content, .{});
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return error.InvalidFormat;

        const benchmarks = root.object.get("benchmarks") orelse return error.InvalidFormat;
        if (benchmarks != .array) return error.InvalidFormat;

        for (benchmarks.array.items) |item| {
            if (item != .object) continue;

            const name_val = item.object.get("name") orelse continue;
            if (name_val != .string) continue;

            const avg_ns = if (item.object.get("avg_ns")) |v| switch (v) {
                .integer => @as(u64, @intCast(v.integer)),
                else => continue,
            } else continue;

            const min_ns = if (item.object.get("min_ns")) |v| switch (v) {
                .integer => @as(u64, @intCast(v.integer)),
                else => avg_ns,
            } else avg_ns;

            const max_ns = if (item.object.get("max_ns")) |v| switch (v) {
                .integer => @as(u64, @intCast(v.integer)),
                else => avg_ns,
            } else avg_ns;

            const iterations = if (item.object.get("iterations")) |v| switch (v) {
                .integer => @as(u64, @intCast(v.integer)),
                else => 1,
            } else 1;

            const timestamp = if (item.object.get("timestamp")) |v| switch (v) {
                .integer => v.integer,
                else => 0,
            } else 0;

            const name_copy = try self.allocator.dupe(u8, name_val.string);
            errdefer self.allocator.free(name_copy);

            try self.baseline.put(self.allocator, name_copy, .{
                .name = name_copy,
                .avg_ns = avg_ns,
                .min_ns = min_ns,
                .max_ns = max_ns,
                .iterations = iterations,
                .timestamp = timestamp,
            });
        }
    }

    /// Benchmark result for comparison
    pub const BenchResult = struct {
        name: []const u8,
        avg_ns: u64,
        min_ns: u64,
        max_ns: u64,
        iterations: u64,
    };

    /// Compare current results against baseline
    pub fn compare(self: *Self, results: []const BenchResult) !RegressionReport {
        var report = RegressionReport.init(self.allocator, self.config);
        errdefer report.deinit();

        // Track which baseline entries we've seen
        var seen = std.StringHashMapUnmanaged(void){};
        defer seen.deinit(self.allocator);

        for (results) |result| {
            const comparison = self.compareOne(result);
            try report.add(comparison);
            try seen.put(self.allocator, result.name, {});
        }

        // Check for missing benchmarks (in baseline but not in current)
        var it = self.baseline.iterator();
        while (it.next()) |entry| {
            if (!seen.contains(entry.key_ptr.*)) {
                try report.add(.{
                    .name = entry.key_ptr.*,
                    .baseline_ns = entry.value_ptr.avg_ns,
                    .current_ns = 0,
                    .change_pct = 0,
                    .status = .missing,
                    .notes = "benchmark removed or renamed",
                });
            }
        }

        return report;
    }

    /// Compare a single benchmark result
    fn compareOne(self: *Self, result: BenchResult) ComparisonResult {
        // Check for unreliable results (high variance)
        const variance = self.calculateCV(result);
        if (variance > self.config.max_cv) {
            return .{
                .name = result.name,
                .baseline_ns = 0,
                .current_ns = result.avg_ns,
                .change_pct = 0,
                .status = .unreliable,
                .notes = "high variance",
            };
        }

        // Check if we have a baseline
        const baseline_entry = self.baseline.get(result.name);
        if (baseline_entry == null) {
            return .{
                .name = result.name,
                .baseline_ns = 0,
                .current_ns = result.avg_ns,
                .change_pct = 0,
                .status = .new,
                .notes = "",
            };
        }

        const baseline = baseline_entry.?;
        const baseline_ns: f64 = @floatFromInt(baseline.avg_ns);
        const current_ns: f64 = @floatFromInt(result.avg_ns);

        // Calculate percentage change
        const change_pct = (current_ns - baseline_ns) / baseline_ns;

        // Determine status
        const status: ComparisonResult.Status = if (@abs(change_pct) < self.config.noise_threshold)
            .stable
        else if (change_pct > self.config.regression_threshold)
            .regression
        else if (change_pct < -self.config.improvement_threshold)
            .improvement
        else
            .stable;

        return .{
            .name = result.name,
            .baseline_ns = baseline.avg_ns,
            .current_ns = result.avg_ns,
            .change_pct = change_pct,
            .status = status,
            .notes = "",
        };
    }

    /// Calculate coefficient of variation from min/max/avg
    fn calculateCV(self: *Self, result: BenchResult) f64 {
        _ = self;
        if (result.avg_ns == 0) return 0;

        const min_f: f64 = @floatFromInt(result.min_ns);
        const max_f: f64 = @floatFromInt(result.max_ns);
        const avg_f: f64 = @floatFromInt(result.avg_ns);

        // Approximate standard deviation from range
        // For normal distribution, range ≈ 4σ for small samples
        const approx_std = (max_f - min_f) / 4.0;
        return approx_std / avg_f;
    }

    /// Save current results as new baseline
    pub fn saveBaseline(self: *Self, results: []const BenchResult) !void {
        const timestamp = std.time.timestamp();

        // Build JSON in memory first
        var output = std.ArrayListUnmanaged(u8){};
        defer output.deinit(self.allocator);

        const w = output.writer(self.allocator);

        try w.writeAll("{\n");
        try w.print("  \"timestamp\": {},\n", .{timestamp});
        try w.print("  \"count\": {},\n", .{results.len});
        try w.writeAll("  \"benchmarks\": [\n");

        for (results, 0..) |r, i| {
            try w.writeAll("    {\n");
            try w.print("      \"name\": \"{s}\",\n", .{r.name});
            try w.print("      \"avg_ns\": {},\n", .{r.avg_ns});
            try w.print("      \"min_ns\": {},\n", .{r.min_ns});
            try w.print("      \"max_ns\": {},\n", .{r.max_ns});
            try w.print("      \"iterations\": {},\n", .{r.iterations});
            try w.print("      \"timestamp\": {}\n", .{timestamp});
            try w.print("    }}{s}\n", .{if (i < results.len - 1) "," else ""});
        }

        try w.writeAll("  ]\n}\n");

        // Write to file
        var file = try std.fs.cwd().createFile(self.config.baseline_path, .{});
        defer file.close();
        try file.writeAll(output.items);
    }

    /// Check if baseline exists
    pub fn hasBaseline(self: Self) bool {
        return self.baseline_loaded and self.baseline.count() > 0;
    }

    /// Get baseline entry count
    pub fn baselineCount(self: Self) usize {
        return self.baseline.count();
    }
};

test "RegressionTracker basic comparison" {
    const allocator = std.testing.allocator;

    var tracker = try RegressionTracker.init(allocator, .{
        .baseline_path = "nonexistent.json",
        .regression_threshold = 0.10,
        .improvement_threshold = 0.10,
    });
    defer tracker.deinit();

    // No baseline, all should be "new"
    const results = [_]RegressionTracker.BenchResult{
        .{ .name = "test1", .avg_ns = 1000, .min_ns = 900, .max_ns = 1100, .iterations = 100 },
    };

    var report = try tracker.compare(&results);
    defer report.deinit();

    try std.testing.expectEqual(@as(usize, 1), report.comparisons.items.len);
    try std.testing.expectEqual(ComparisonResult.Status.new, report.comparisons.items[0].status);
    try std.testing.expect(!report.hasRegressions());
}

test "RegressionConfig defaults" {
    const config = RegressionConfig{};
    try std.testing.expectEqual(@as(f64, 0.10), config.regression_threshold);
    try std.testing.expectEqual(@as(f64, 0.10), config.improvement_threshold);
    try std.testing.expectEqual(@as(f64, 0.03), config.noise_threshold);
}

test "RegressionReport status counting" {
    const allocator = std.testing.allocator;

    var report = RegressionReport.init(allocator, .{});
    defer report.deinit();

    try report.add(.{
        .name = "test1",
        .baseline_ns = 1000,
        .current_ns = 1500,
        .change_pct = 0.50,
        .status = .regression,
        .notes = "",
    });

    try report.add(.{
        .name = "test2",
        .baseline_ns = 1000,
        .current_ns = 500,
        .change_pct = -0.50,
        .status = .improvement,
        .notes = "",
    });

    try report.add(.{
        .name = "test3",
        .baseline_ns = 1000,
        .current_ns = 1000,
        .change_pct = 0.0,
        .status = .stable,
        .notes = "",
    });

    try std.testing.expectEqual(@as(usize, 1), report.countByStatus(.regression));
    try std.testing.expectEqual(@as(usize, 1), report.countByStatus(.improvement));
    try std.testing.expectEqual(@as(usize, 1), report.countByStatus(.stable));
    try std.testing.expect(report.hasRegressions());
}
