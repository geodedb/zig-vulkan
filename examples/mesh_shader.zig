//! Mesh Shader Capabilities Example
//!
//! Demonstrates querying mesh shader capabilities on Vulkan devices.
//! Mesh shaders replace the traditional vertex/geometry pipeline with
//! a more flexible compute-like approach to geometry processing.
//!
//! ## Hardware Requirements
//!
//! - NVIDIA: Turing (RTX 20-series) or newer
//! - AMD: RDNA2 (RX 6000-series) or newer
//! - Intel: Arc GPUs (Alchemist+)
//!
//! ## Vulkan Extensions
//!
//! - VK_EXT_mesh_shader (newer, recommended)
//! - VK_NV_mesh_shader (NVIDIA-specific, older)
//!
//! ## Pipeline Stages
//!
//! Traditional: Vertex -> Tessellation -> Geometry -> Rasterizer
//! Mesh Shader: Task Shader (optional) -> Mesh Shader -> Rasterizer
//!
//! ## Usage
//!
//! ```bash
//! zig build example-mesh-shader
//! ./zig-out/bin/example-mesh-shader
//! ```
//!
//! ## Note
//!
//! This library focuses on compute shaders. Mesh shader queries are for
//! capability detection only. Full mesh shader pipeline support would
//! require graphics pipeline infrastructure.

const std = @import("std");
const vulkan = @import("vulkan");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Mesh Shader Capabilities Example\n", .{});
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

    // Check each device for mesh shader support
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

        // Query mesh shader capabilities
        const mesh_caps = vulkan.queryMeshShaderCapabilities(&ctx, allocator);

        if (!mesh_caps.isAvailable()) {
            std.debug.print("  Mesh Shaders: NOT SUPPORTED\n", .{});
            std.debug.print("  - Requires Turing (NVIDIA), RDNA2 (AMD), or Arc (Intel)\n\n", .{});
            continue;
        }

        std.debug.print("  Mesh Shaders: SUPPORTED\n", .{});

        // Print features
        std.debug.print("\n  Features:\n", .{});
        std.debug.print("    Task Shader:         {s}\n", .{if (mesh_caps.hasTaskShaders()) "YES" else "NO"});
        std.debug.print("    Mesh Shader:         {s}\n", .{if (mesh_caps.mesh_shader) "YES" else "NO"});
        std.debug.print("    Multiview:           {s}\n", .{if (mesh_caps.multiview_mesh_shader) "YES" else "NO"});
        std.debug.print("    Primitive Shading:   {s}\n", .{if (mesh_caps.primitive_fragment_shading_rate_mesh_shader) "YES" else "NO"});

        // Print limits
        std.debug.print("\n  Mesh Shader Limits:\n", .{});
        std.debug.print("    Max Output Vertices:    {}\n", .{mesh_caps.max_mesh_output_vertices});
        std.debug.print("    Max Output Primitives:  {}\n", .{mesh_caps.max_mesh_output_primitives});
        std.debug.print("    Max Mesh Workgroup Count: {}x{}x{}\n", .{
            mesh_caps.max_mesh_work_group_count[0],
            mesh_caps.max_mesh_work_group_count[1],
            mesh_caps.max_mesh_work_group_count[2],
        });
        std.debug.print("    Max Mesh Workgroup Invocations: {}\n", .{mesh_caps.max_mesh_work_group_invocations});
        std.debug.print("    Preferred Mesh Workgroup Size:  {}\n", .{mesh_caps.recommendedMeshWorkGroupSize()});

        if (mesh_caps.hasTaskShaders()) {
            std.debug.print("\n  Task Shader Limits:\n", .{});
            std.debug.print("    Max Task Workgroup Count: {}x{}x{}\n", .{
                mesh_caps.max_task_work_group_count[0],
                mesh_caps.max_task_work_group_count[1],
                mesh_caps.max_task_work_group_count[2],
            });
            std.debug.print("    Max Task Workgroup Invocations: {}\n", .{mesh_caps.max_task_work_group_invocations});
            std.debug.print("    Preferred Task Workgroup Size:  {}\n", .{mesh_caps.preferred_task_work_group_invocations});
        }

        std.debug.print("\n", .{});
    }

    // Show example mesh shader code
    printMeshShaderExample();
}

fn printMeshShaderExample() void {
    std.debug.print(
        \\Mesh Shader Example (VK_EXT_mesh_shader):
        \\==========================================
        \\
        \\Mesh shaders combine vertex and primitive assembly into one stage,
        \\providing more control over geometry processing.
        \\
        \\// Task Shader (optional, for coarse culling/LOD)
        \\#version 460
        \\#extension GL_EXT_mesh_shader : require
        \\
        \\layout(local_size_x = 32) in;
        \\
        \\// Emit mesh shader workgroups
        \\taskPayloadSharedEXT MeshPayload {{
        \\    uint meshletIndices[32];
        \\}} payload;
        \\
        \\void main() {{
        \\    uint tid = gl_LocalInvocationID.x;
        \\
        \\    // Frustum/occlusion cull meshlets here
        \\    if (isVisible(tid)) {{
        \\        payload.meshletIndices[tid] = tid;
        \\    }}
        \\
        \\    // Emit mesh shader workgroups
        \\    EmitMeshTasksEXT(visibleCount, 1, 1);
        \\}}
        \\
        \\// Mesh Shader
        \\#version 460
        \\#extension GL_EXT_mesh_shader : require
        \\
        \\layout(local_size_x = 32) in;
        \\layout(triangles, max_vertices = 64, max_primitives = 126) out;
        \\
        \\// Output vertex attributes
        \\layout(location = 0) out vec3 outNormal[];
        \\layout(location = 1) out vec2 outUV[];
        \\
        \\// Meshlet data (pre-computed)
        \\struct Meshlet {{
        \\    uint vertexOffset;
        \\    uint triangleOffset;
        \\    uint vertexCount;
        \\    uint triangleCount;
        \\}};
        \\
        \\void main() {{
        \\    uint meshletIndex = gl_WorkGroupID.x;
        \\    uint tid = gl_LocalInvocationID.x;
        \\
        \\    Meshlet meshlet = meshlets[meshletIndex];
        \\
        \\    // Set output counts
        \\    SetMeshOutputsEXT(meshlet.vertexCount, meshlet.triangleCount);
        \\
        \\    // Emit vertices (parallel)
        \\    if (tid < meshlet.vertexCount) {{
        \\        uint vi = meshlet.vertexOffset + tid;
        \\        vec3 pos = positions[vi];
        \\        gl_MeshVerticesEXT[tid].gl_Position = mvp * vec4(pos, 1.0);
        \\        outNormal[tid] = normals[vi];
        \\        outUV[tid] = uvs[vi];
        \\    }}
        \\
        \\    // Emit primitives (parallel)
        \\    if (tid < meshlet.triangleCount) {{
        \\        uint ti = meshlet.triangleOffset + tid * 3;
        \\        gl_PrimitiveTriangleIndicesEXT[tid] = uvec3(
        \\            indices[ti], indices[ti+1], indices[ti+2]
        \\        );
        \\    }}
        \\}}
        \\
        \\Benefits of Mesh Shaders:
        \\- Flexible workgroup-based geometry processing
        \\- Efficient meshlet-based rendering for complex scenes
        \\- Built-in culling before rasterization
        \\- Shared memory for intermediate calculations
        \\- Better GPU utilization for geometry-heavy workloads
        \\
        \\Note: This library provides capability queries only.
        \\Full mesh shader support requires graphics pipeline infrastructure.
        \\
    , .{});
}
