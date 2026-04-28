//! observability.zig -- single-file otel seam for vulkan.
//!
//! When `build_options.with_otel == false`, every public function in this
//! file collapses to a comptime-known no-op so the optimizer eliminates
//! the call site entirely. When true, calls forward to the appropriate
//! `otel.api.trace` / `otel.instrumentation.http` helper.
//!
//! INVARIANT: every emit is `catch {}`-discarded. A failing exporter
//! never reaches the caller -- telemetry must never break Vulkan
//! operations (queue submits, device waits, pipeline creation).
//!
//! NOTE: deviation from template -- vulkan operations don't fit the
//! HTTP `instrumentation.http.startClientSpan` shape (HTTP method/url
//! is lossy for a GPU compute submission). We expose the generic
//! `startClientSpan` / `startServerSpan` helpers for symmetry with
//! grpc-zig (and to keep the seam template-compatible), AND a
//! gpu-specific `startVulkanSpan(operation)` helper that constructs a
//! span via `otel.api.trace.Tracer.startSpan` directly with a
//! gpu-shaped span name (`"vulkan.{operation}"`) and the
//! `gpu.system="vulkan"` / `gpu.operation` semantic conventions.

const std = @import("std");
const build_options = @import("build_options");

pub const enabled: bool = build_options.with_otel;

const otel = if (enabled) @import("otel") else struct {};

/// Opaque handle for an in-flight operation. When disabled, this is a
/// zero-sized struct the optimizer drops entirely. The no-op fallback's
/// `end` signature MUST match `otel.api.trace.Span.end(?u64) void` so
/// callers can write `span.end(null)` regardless of `enabled` state.
///
/// The stub also exposes `setStringAttribute` / `setIntAttribute` no-op
/// shims so call sites can attach domain attributes (e.g.
/// `vulkan.queue_family`, `vulkan.cmd_buffer_count`) without conditional
/// compilation. Both shims swallow the error union (`!void`) returned
/// by the real otel API so the call shape `span.setX(...) catch {}`
/// works under both flag values.
pub const Span = if (enabled) otel.api.trace.Span else struct {
    pub fn end(self: @This(), _: ?u64) void {
        _ = self;
    }
    pub fn setStringAttribute(self: @This(), _: []const u8, _: []const u8) !void {
        _ = self;
    }
    pub fn setIntAttribute(self: @This(), _: []const u8, _: i64) !void {
        _ = self;
    }
};

/// Helper: reads either a `[]const u8` field OR a function-typed field
/// returning `[]const u8`, so the seam accepts both shapes -- the closure
/// shape the recipe template ships AND the plain-string-field shape used
/// at hot-path call sites.
inline fn accessOr(req: anytype, comptime field: []const u8) []const u8 {
    const T = @TypeOf(@field(req, field));
    return switch (@typeInfo(T)) {
        .pointer => @field(req, field),
        .@"fn" => @field(req, field)(),
        else => @compileError("observability: unsupported field type for " ++ field),
    };
}

/// Start a span for a generic outbound operation. Kept for template
/// symmetry; vulkan hot paths use `startVulkanSpan` instead.
pub fn startClientSpan(req: anytype) Span {
    if (!enabled) return Span{};
    const tracer = global.tracer orelse return Span{};
    const span = otel.instrumentation.http.startClientSpan(
        tracer.*,
        .empty(),
        .{
            .method = accessOr(req, "method"),
            .url = accessOr(req, "url"),
        },
    ) catch return Span{};
    return span;
}

pub fn endClientSpan(span: Span, status: u16) void {
    if (!enabled) return;
    otel.instrumentation.http.endClientSpan(span, .{ .status_code = status });
}

/// Start a span for a generic inbound operation. Kept for template
/// symmetry; vulkan has no authoritative-server entry point.
pub fn startServerSpan(req: anytype) Span {
    if (!enabled) return Span{};
    const tracer = global.tracer orelse return Span{};
    const span = otel.instrumentation.http.startServerSpan(
        tracer.*,
        .empty(),
        .{
            .method = accessOr(req, "method"),
            .route = accessOr(req, "route"),
            .url = accessOr(req, "url"),
        },
    ) catch return Span{};
    return span;
}

pub fn endServerSpan(span: Span, status: u16) void {
    if (!enabled) return;
    otel.instrumentation.http.endServerSpan(span, status);
}

/// Start a span for a Vulkan GPU operation.
///
/// Span shape:
///   - name: `"vulkan.{operation}"` (e.g. `"vulkan.queue_submit"`,
///     `"vulkan.device_wait_idle"`, `"vulkan.pipeline_create"`)
///   - kind: `.client` (the application acts as a client to the GPU
///     driver/queue; the OpenTelemetry `gpu.*` semantic conventions
///     specify `client` for the calling side of a GPU interaction).
///   - attributes:
///       * `gpu.system = "vulkan"`
///       * `gpu.operation = operation`
///
/// `operation` MUST be a string literal or a slice whose lifetime
/// outlives the span. The SDK dupes the span name into the span arena,
/// but the `gpu.operation` attribute is stored by reference at the
/// stringified value layer; using a stable string keeps the recorder
/// safe from dangling slices.
///
/// Use `endVulkanSpan(span, ok)` to close. Caller-supplied attributes
/// (e.g. `vulkan.queue_family`, `vulkan.cmd_buffer_count`) can be
/// attached to the returned span via `span.setStringAttribute` /
/// `span.setIntAttribute` between start and end -- those calls are also
/// `catch {}`-discarded at the call site to preserve the non-fatal
/// telemetry invariant.
pub fn startVulkanSpan(operation: []const u8) Span {
    if (!enabled) return Span{};
    const tracer = global.tracer orelse return Span{};
    // Compose the span name on the stack; the SDK dupes the name into
    // the span arena, so a pointer to a stack buffer is safe here.
    var name_buf: [128]u8 = undefined;
    const span_name = std.fmt.bufPrint(
        name_buf[0..],
        "vulkan.{s}",
        .{operation},
    ) catch "vulkan";
    const span = tracer.startSpan(.empty(), span_name, .{ .kind = .client }) catch return Span{};
    // Best-effort attribute attachment. Failures (OOM in the attribute
    // set) cannot block the GPU operation -- telemetry is non-fatal.
    span.setStringAttribute("gpu.system", "vulkan") catch {};
    span.setStringAttribute("gpu.operation", operation) catch {};
    return span;
}

/// Close a Vulkan operation span.
///
/// On `ok = true` we set status `.ok`; on `ok = false` we set status
/// `.error_` with an empty description (callers wanting a richer error
/// message can attach attributes/events on the span before calling end).
/// `setStatus` + `end` are idempotent for status reporting -- only the
/// first status write wins, so a deferred `span.end(null)` at the call
/// site remains safe.
pub fn endVulkanSpan(span: Span, ok: bool) void {
    if (!enabled) return;
    if (ok) {
        span.setStatus(.ok, "");
    } else {
        // `StatusCode` uses `@"error"` (escaped keyword) for the
        // failure variant -- see otel/src/api/trace.zig:35.
        span.setStatus(.@"error", "");
    }
    span.end(null);
}

/// Set once at process start by `observability_init.Otel.installGlobals`.
/// Only referenced when `enabled == true`; otherwise the optimizer drops
/// the whole struct.
pub const global = if (enabled) struct {
    pub var tracer: ?*otel.api.trace.Tracer = null;
    pub var meter: ?*otel.api.metrics.Meter = null;
} else struct {};
