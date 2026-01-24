//! Vulkan Extensions Query Example
//!
//! Demonstrates querying available Vulkan extensions and device capabilities:
//! - Subgroup operations
//! - Descriptor indexing (bindless resources)
//! - Ray tracing
//! - Video encode/decode
//! - Cooperative matrices

const std = @import("std");
const vulkan = @import("vulkan");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Vulkan Extensions Query\n", .{});
    std.debug.print("=======================\n\n", .{});

    // Check availability
    if (!vulkan.isAvailable()) {
        std.debug.print("Vulkan not available\n", .{});
        return;
    }

    // Create context
    var ctx = vulkan.Context.init(allocator, .{
        .app_name = "extensions_example",
        .min_api_version = vulkan.vk.API_VERSION_1_1, // Require Vulkan 1.1+
    }) catch |e| {
        std.debug.print("Failed to create context: {s}\n", .{vulkan.vk.describe(e)});
        return;
    };
    defer ctx.deinit();

    const device_info = ctx.deviceInfo();
    std.debug.print("Device: {s}\n", .{device_info.getName()});
    std.debug.print("API Version: {}.{}.{}\n\n", .{
        vulkan.vk.apiVersionMajor(device_info.api_version),
        vulkan.vk.apiVersionMinor(device_info.api_version),
        vulkan.vk.apiVersionPatch(device_info.api_version),
    });

    // Query all extension capabilities
    const caps = vulkan.queryAllCapabilities(&ctx, allocator);

    // Print subgroup capabilities
    std.debug.print("Subgroup Operations:\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});
    if (caps.subgroup) |sg| {
        std.debug.print("  Subgroup size: {}\n", .{sg.subgroup_size});
        std.debug.print("  Compute support: {}\n", .{sg.supportsCompute()});
        std.debug.print("  Operations:\n", .{});
        std.debug.print("    Basic:     {}\n", .{sg.supportsOperation(.basic)});
        std.debug.print("    Vote:      {}\n", .{sg.supportsOperation(.vote)});
        std.debug.print("    Arithmetic:{}\n", .{sg.supportsOperation(.arithmetic)});
        std.debug.print("    Ballot:    {}\n", .{sg.supportsOperation(.ballot)});
        std.debug.print("    Shuffle:   {}\n", .{sg.supportsOperation(.shuffle)});
        std.debug.print("    Clustered: {}\n", .{sg.supportsOperation(.clustered)});
        std.debug.print("    Quad:      {}\n", .{sg.supportsOperation(.quad)});
    } else {
        std.debug.print("  Not available (requires Vulkan 1.1+)\n", .{});
    }
    std.debug.print("\n", .{});

    // Print descriptor indexing capabilities
    std.debug.print("Descriptor Indexing (Bindless):\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});
    if (caps.descriptor_indexing) |di| {
        std.debug.print("  Bindless support: {}\n", .{di.supportsBindless()});
        std.debug.print("  Runtime arrays:   {}\n", .{di.runtime_descriptor_array});
        std.debug.print("  Partially bound:  {}\n", .{di.partially_bound});
        std.debug.print("  Variable count:   {}\n", .{di.variable_descriptor_count});
        std.debug.print("  Update after bind:\n", .{});
        std.debug.print("    Storage buffers: {}\n", .{di.storage_buffer_update_after_bind});
        std.debug.print("    Sampled images:  {}\n", .{di.sampled_image_update_after_bind});
        std.debug.print("  Non-uniform indexing:\n", .{});
        std.debug.print("    Storage buffers: {}\n", .{di.storage_buffer_non_uniform_indexing});
        std.debug.print("    Sampled images:  {}\n", .{di.sampled_image_non_uniform_indexing});
    } else {
        std.debug.print("  Not available (requires VK_EXT_descriptor_indexing)\n", .{});
    }
    std.debug.print("\n", .{});

    // Print ray tracing capabilities
    std.debug.print("Ray Tracing (VK_KHR_ray_tracing_pipeline):\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});
    if (caps.ray_tracing) |rt| {
        if (rt.isAvailable()) {
            std.debug.print("  Ray tracing pipeline: {}\n", .{rt.ray_tracing_pipeline});
            std.debug.print("  Max recursion depth:  {}\n", .{rt.max_ray_recursion_depth});
            std.debug.print("  Indirect trace rays:  {}\n", .{rt.trace_rays_indirect});
            std.debug.print("  Primitive culling:    {}\n", .{rt.ray_traversal_primitive_culling});
            std.debug.print("  Shader group handle:  {} bytes\n", .{rt.shader_group_handle_size});
        } else {
            std.debug.print("  Not supported on this device\n", .{});
        }
    } else {
        std.debug.print("  Not available (requires VK_KHR_ray_tracing_pipeline)\n", .{});
    }
    std.debug.print("\n", .{});

    // Print video capabilities
    std.debug.print("Video Queue (VK_KHR_video_queue):\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});
    if (caps.video) |vid| {
        if (vid.isAvailable()) {
            std.debug.print("  Queue family: {}\n", .{vid.video_queue_family.?});
            std.debug.print("  Encode H.264: {}\n", .{vid.encode_h264});
            std.debug.print("  Encode H.265: {}\n", .{vid.encode_h265});
            std.debug.print("  Decode H.264: {}\n", .{vid.decode_h264});
            std.debug.print("  Decode H.265: {}\n", .{vid.decode_h265});
            std.debug.print("  Decode AV1:   {}\n", .{vid.decode_av1});
        } else {
            std.debug.print("  Not supported on this device\n", .{});
        }
    } else {
        std.debug.print("  Not available (requires VK_KHR_video_queue)\n", .{});
    }
    std.debug.print("\n", .{});

    // Print cooperative matrix capabilities
    std.debug.print("Cooperative Matrices (VK_KHR_cooperative_matrix):\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});
    if (caps.cooperative_matrix) |cm| {
        if (cm.isAvailable()) {
            std.debug.print("  Cooperative matrix:       {}\n", .{cm.cooperative_matrix});
            std.debug.print("  Robust buffer access:     {}\n", .{cm.robust_buffer_access});
            std.debug.print("  Supported configurations: {}\n", .{cm.configurations.len});
        } else {
            std.debug.print("  Not supported on this device\n", .{});
        }
    } else {
        std.debug.print("  Not available (requires VK_KHR_cooperative_matrix)\n", .{});
    }
    std.debug.print("\n", .{});

    // Summary
    std.debug.print("Summary:\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    var supported_count: u32 = 0;
    if (caps.subgroup != null) supported_count += 1;
    if (caps.descriptor_indexing != null) supported_count += 1;
    if (caps.ray_tracing != null and caps.ray_tracing.?.isAvailable()) supported_count += 1;
    if (caps.video != null and caps.video.?.isAvailable()) supported_count += 1;
    if (caps.cooperative_matrix != null and caps.cooperative_matrix.?.isAvailable()) supported_count += 1;

    std.debug.print("  Extensions queried: 5\n", .{});
    std.debug.print("  Extensions available: {}\n", .{supported_count});

    if (caps.subgroup) |sg| {
        if (sg.supportsOperation(.arithmetic)) {
            std.debug.print("\n  Recommendation: Use subgroup arithmetic for reductions!\n", .{});
        }
    }

    if (caps.descriptor_indexing) |di| {
        if (di.supportsBindless()) {
            std.debug.print("  Recommendation: Use bindless descriptors for flexibility!\n", .{});
        }
    }

    std.debug.print("\nExample completed.\n", .{});
}
