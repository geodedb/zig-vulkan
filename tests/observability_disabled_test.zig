//! Verifies the no-op invariant when -Dwith_otel=false (the default).
//!
//! vulkan publishes its public surface as the "vulkan" module rooted at
//! src/lib.zig. The seam is exposed as `vulkan.observability` so this
//! test can drive the disabled-mode invariants through the public root.

const std = @import("std");
const vulkan = @import("vulkan");

test "build_options.with_otel exists and defaults to false" {
    const build_options = @import("build_options");
    try std.testing.expect(@hasDecl(build_options, "with_otel"));
    // The "default" assertion only holds for the disabled build. When
    // the operator forces -Dwith_otel=true, this test runs against the
    // enabled module and must skip rather than misreport the default.
    if (build_options.with_otel) return error.SkipZigTest;
    try std.testing.expectEqual(false, build_options.with_otel);
}

test "observability.enabled tracks build_options.with_otel" {
    const build_options = @import("build_options");
    try std.testing.expectEqual(build_options.with_otel, vulkan.observability.enabled);
}

test "client/server seams are no-ops when otel disabled" {
    if (vulkan.observability.enabled) return error.SkipZigTest;
    var span = vulkan.observability.startClientSpan(.{
        .method = "GET",
        .route = "/",
        .url = "https://example.test/",
    });
    vulkan.observability.endClientSpan(span, 200);
    // Belt-and-suspenders: the seam's `endClientSpan` already covers
    // the lifecycle; this line just verifies the no-op fallback's
    // signature matches the real API's `Span.end(?u64) void`.
    span.end(null);

    var sspan = vulkan.observability.startServerSpan(.{
        .method = "GET",
        .route = "/",
        .url = "/",
    });
    vulkan.observability.endServerSpan(sspan, 200);
    sspan.end(null);
}

test "vulkan span helper is a no-op when otel disabled" {
    if (vulkan.observability.enabled) return error.SkipZigTest;
    var span = vulkan.observability.startVulkanSpan("queue_submit");
    vulkan.observability.endVulkanSpan(span, true);
    span.end(null);
}
