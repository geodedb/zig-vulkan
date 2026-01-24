//! Ray Tracing Capabilities Example
//!
//! Demonstrates querying ray tracing hardware capabilities on Vulkan devices.
//! Ray tracing acceleration is available on modern GPUs with dedicated RT cores.
//!
//! ## Hardware Requirements
//!
//! - NVIDIA: RTX 20-series or newer (Turing+)
//! - AMD: RX 6000-series or newer (RDNA2+)
//! - Intel: Arc GPUs (Alchemist+)
//!
//! ## Vulkan Extensions
//!
//! Ray tracing requires the following extensions:
//! - VK_KHR_acceleration_structure - BVH construction
//! - VK_KHR_ray_tracing_pipeline - Ray tracing shaders
//! - VK_KHR_ray_query - Inline ray queries in any shader
//! - VK_KHR_deferred_host_operations - Async operations
//!
//! ## Usage
//!
//! ```bash
//! zig build example-ray-tracing
//! ./zig-out/bin/example-ray-tracing
//! ```
//!
//! ## Note
//!
//! This library focuses on compute shaders. Ray tracing queries are for
//! capability detection only. Full ray tracing pipeline support would require
//! additional graphics pipeline infrastructure.

const std = @import("std");
const vulkan = @import("vulkan");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Ray Tracing Capabilities Example\n", .{});
    std.debug.print("=================================\n\n", .{});

    // Check Vulkan availability
    if (!vulkan.isAvailable()) {
        std.debug.print("Vulkan is not available on this system.\n", .{});
        return;
    }

    // Probe for devices
    var probe = vulkan.probe(allocator);
    defer probe.deinit(allocator);

    if (!probe.isAvailable()) {
        std.debug.print("No Vulkan devices found.\n", .{});
        return;
    }

    std.debug.print("Found {} Vulkan device(s)\n\n", .{probe.devices.len});

    // Check each device for ray tracing support
    for (probe.devices, 0..) |device, i| {
        std.debug.print("Device {}: {s}\n", .{ i, device.getName() });
        std.debug.print("  Type: {s}\n", .{@tagName(device.device_type)});

        // Create context to query capabilities
        var ctx = vulkan.Context.init(allocator, .{
            .device_index = @intCast(i),
        }) catch |e| {
            std.debug.print("  Failed to create context: {}\n\n", .{e});
            continue;
        };
        defer ctx.deinit();

        // Query ray tracing capabilities
        const rt_caps = vulkan.queryRayTracingCapabilities(&ctx) catch |e| {
            std.debug.print("  Failed to query capabilities: {}\n\n", .{e});
            continue;
        };

        if (!rt_caps.isAvailable()) {
            std.debug.print("  Ray Tracing: NOT SUPPORTED\n", .{});
            std.debug.print("  - Hardware may lack RT cores\n", .{});
            std.debug.print("  - Or driver may not support ray tracing extensions\n\n", .{});
            continue;
        }

        std.debug.print("  Ray Tracing: SUPPORTED\n", .{});

        // Print supported features
        std.debug.print("\n  Features:\n", .{});
        std.debug.print("    Ray Tracing Pipeline:    {s}\n", .{if (rt_caps.ray_tracing_pipeline) "YES" else "NO"});
        std.debug.print("    Trace Rays Indirect:     {s}\n", .{if (rt_caps.trace_rays_indirect) "YES" else "NO"});
        std.debug.print("    Primitive Culling:       {s}\n", .{if (rt_caps.ray_traversal_primitive_culling) "YES" else "NO"});

        // Print limits
        std.debug.print("\n  Limits:\n", .{});
        std.debug.print("    Max Ray Recursion Depth:       {}\n", .{rt_caps.max_ray_recursion_depth});
        std.debug.print("    Shader Group Handle Size:      {} bytes\n", .{rt_caps.shader_group_handle_size});
        std.debug.print("    Max Ray Dispatch Invocations:  {}\n", .{rt_caps.max_ray_dispatch_invocation_count});

        std.debug.print("\n", .{});
    }

    // Show example usage with ray query in compute shader
    printRayQueryExample();
}

fn printRayQueryExample() void {
    std.debug.print(
        \\Ray Query in Compute Shader (VK_KHR_ray_query):
        \\=================================================
        \\
        \\Ray query allows tracing rays from any shader stage, including compute.
        \\This is useful for visibility testing, ambient occlusion, and more.
        \\
        \\#version 460
        \\#extension GL_EXT_ray_query : require
        \\
        \\layout(local_size_x = 8, local_size_y = 8) in;
        \\
        \\layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
        \\layout(set = 0, binding = 1, rgba8) writeonly uniform image2D outputImage;
        \\
        \\void main() {{
        \\    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
        \\
        \\    // Setup ray
        \\    vec3 origin = vec3(0.0, 0.0, -5.0);
        \\    vec3 direction = normalize(vec3(pixel, 0.0) - origin);
        \\    float tMin = 0.001;
        \\    float tMax = 1000.0;
        \\
        \\    // Initialize ray query
        \\    rayQueryEXT rayQuery;
        \\    rayQueryInitializeEXT(
        \\        rayQuery,
        \\        topLevelAS,
        \\        gl_RayFlagsOpaqueEXT,
        \\        0xFF,        // Cull mask
        \\        origin,
        \\        tMin,
        \\        direction,
        \\        tMax
        \\    );
        \\
        \\    // Traverse acceleration structure
        \\    while (rayQueryProceedEXT(rayQuery)) {{
        \\        // Optional: Handle non-opaque geometry
        \\        if (rayQueryGetIntersectionTypeEXT(rayQuery, false) ==
        \\            gl_RayQueryCandidateIntersectionTriangleEXT) {{
        \\            rayQueryConfirmIntersectionEXT(rayQuery);
        \\        }}
        \\    }}
        \\
        \\    // Check result
        \\    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);  // Miss = black
        \\    if (rayQueryGetIntersectionTypeEXT(rayQuery, true) !=
        \\        gl_RayQueryCommittedIntersectionNoneEXT) {{
        \\        // Hit - shade based on distance
        \\        float t = rayQueryGetIntersectionTEXT(rayQuery, true);
        \\        color = vec4(1.0 / (1.0 + t * 0.1));
        \\    }}
        \\
        \\    imageStore(outputImage, pixel, color);
        \\}}
        \\
        \\Note: This library provides capability queries only.
        \\Full ray tracing requires acceleration structure management
        \\which is beyond the current compute-focused scope.
        \\
    , .{});
}
