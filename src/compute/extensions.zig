// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Vulkan Extensions Support
//!
//! High-level wrappers for optional Vulkan extensions:
//! - Subgroup operations (Vulkan 1.1+) for SIMD-style parallelism
//! - Descriptor indexing for bindless resource access
//! - Ray tracing pipeline for hardware ray tracing
//! - Video queue for hardware video decode/encode
//! - Cooperative matrices for tensor operations
//!
//! ## Subgroup Operations
//!
//! ```zig
//! const ext = @import("vulkan").extensions;
//!
//! // Query subgroup capabilities
//! const props = try ext.querySubgroupProperties(&ctx);
//! std.debug.print("Subgroup size: {}\n", .{props.subgroup_size});
//!
//! if (props.supported_operations.arithmetic) {
//!     // Use subgroup arithmetic operations in shaders
//! }
//! ```
//!
//! ## Descriptor Indexing
//!
//! ```zig
//! // Check for bindless descriptor support
//! const features = try ext.queryDescriptorIndexingFeatures(&ctx);
//! if (features.runtime_descriptor_array != 0) {
//!     // Create bindless descriptor layouts
//! }
//! ```

const std = @import("std");
const types = @import("../vk/types.zig");
const loader = @import("../vk/loader.zig");
const err = @import("../vk/error.zig");
const Context = @import("context.zig").Context;

// =============================================================================
// Subgroup Operations
// =============================================================================

/// Subgroup capabilities for a device
pub const SubgroupCapabilities = struct {
    /// Size of subgroups (typically 32 for NVIDIA, 64 for AMD)
    subgroup_size: u32,
    /// Shader stages that support subgroup operations
    supported_stages: types.ShaderStageFlags,
    /// Supported subgroup operations
    supported_operations: types.SubgroupFeatureFlags,
    /// Whether quad operations work in all stages
    quad_operations_in_all_stages: bool,

    /// Check if a specific operation is supported
    pub fn supportsOperation(self: SubgroupCapabilities, op: SubgroupOperation) bool {
        return switch (op) {
            .basic => self.supported_operations.basic,
            .vote => self.supported_operations.vote,
            .arithmetic => self.supported_operations.arithmetic,
            .ballot => self.supported_operations.ballot,
            .shuffle => self.supported_operations.shuffle,
            .shuffle_relative => self.supported_operations.shuffle_relative,
            .clustered => self.supported_operations.clustered,
            .quad => self.supported_operations.quad,
        };
    }

    /// Check if subgroups are supported in compute shaders
    pub fn supportsCompute(self: SubgroupCapabilities) bool {
        return self.supported_stages.compute;
    }
};

/// Subgroup operation types
pub const SubgroupOperation = enum {
    basic,
    vote,
    arithmetic,
    ballot,
    shuffle,
    shuffle_relative,
    clustered,
    quad,
};

/// Query subgroup properties from a context
pub fn querySubgroupProperties(ctx: *Context) err.Error!SubgroupCapabilities {
    var props = types.PhysicalDeviceSubgroupProperties{};

    // Need to chain with PhysicalDeviceProperties2
    var props2 = types.PhysicalDeviceProperties2{
        .s_type = .physical_device_properties_2,
        .p_next = &props,
    };

    if (ctx.instance_dispatch.vkGetPhysicalDeviceProperties2) |getProps2| {
        getProps2(ctx.physical_device, &props2);

        return SubgroupCapabilities{
            .subgroup_size = props.subgroup_size,
            .supported_stages = props.supported_stages,
            .supported_operations = props.supported_operations,
            .quad_operations_in_all_stages = props.quad_operations_in_all_stages != 0,
        };
    }

    return err.Error.ExtensionNotPresent;
}

// =============================================================================
// Descriptor Indexing
// =============================================================================

/// Descriptor indexing capabilities
pub const DescriptorIndexingCapabilities = struct {
    /// Runtime descriptor arrays supported
    runtime_descriptor_array: bool,
    /// Update after bind for storage buffers
    storage_buffer_update_after_bind: bool,
    /// Update after bind for sampled images
    sampled_image_update_after_bind: bool,
    /// Partially bound descriptors
    partially_bound: bool,
    /// Variable descriptor count
    variable_descriptor_count: bool,
    /// Non-uniform indexing for storage buffers
    storage_buffer_non_uniform_indexing: bool,
    /// Non-uniform indexing for sampled images
    sampled_image_non_uniform_indexing: bool,

    /// Check if bindless resources are fully supported
    pub fn supportsBindless(self: DescriptorIndexingCapabilities) bool {
        return self.runtime_descriptor_array and
            self.partially_bound and
            self.variable_descriptor_count;
    }
};

/// Query descriptor indexing features
pub fn queryDescriptorIndexingFeatures(ctx: *Context) err.Error!DescriptorIndexingCapabilities {
    var features = types.PhysicalDeviceDescriptorIndexingFeatures{};

    var features2 = types.PhysicalDeviceFeatures2{
        .p_next = &features,
    };

    if (ctx.instance_dispatch.vkGetPhysicalDeviceFeatures2) |getFeatures2| {
        getFeatures2(ctx.physical_device, &features2);

        return DescriptorIndexingCapabilities{
            .runtime_descriptor_array = features.runtime_descriptor_array != 0,
            .storage_buffer_update_after_bind = features.descriptor_binding_storage_buffer_update_after_bind != 0,
            .sampled_image_update_after_bind = features.descriptor_binding_sampled_image_update_after_bind != 0,
            .partially_bound = features.descriptor_binding_partially_bound != 0,
            .variable_descriptor_count = features.descriptor_binding_variable_descriptor_count != 0,
            .storage_buffer_non_uniform_indexing = features.shader_storage_buffer_array_non_uniform_indexing != 0,
            .sampled_image_non_uniform_indexing = features.shader_sampled_image_array_non_uniform_indexing != 0,
        };
    }

    return err.Error.ExtensionNotPresent;
}

// =============================================================================
// Ray Tracing
// =============================================================================

/// Ray tracing capabilities
pub const RayTracingCapabilities = struct {
    /// Ray tracing pipeline is supported
    ray_tracing_pipeline: bool,
    /// Indirect trace rays supported
    trace_rays_indirect: bool,
    /// Primitive culling during traversal
    ray_traversal_primitive_culling: bool,
    /// Maximum recursion depth
    max_ray_recursion_depth: u32,
    /// Shader group handle size
    shader_group_handle_size: u32,
    /// Maximum ray dispatch invocation count
    max_ray_dispatch_invocation_count: u32,

    /// Check if basic ray tracing is available
    pub fn isAvailable(self: RayTracingCapabilities) bool {
        return self.ray_tracing_pipeline;
    }
};

/// Query ray tracing pipeline features and properties
pub fn queryRayTracingCapabilities(ctx: *Context) err.Error!RayTracingCapabilities {
    var features = types.PhysicalDeviceRayTracingPipelineFeaturesKHR{};
    var props = types.PhysicalDeviceRayTracingPipelinePropertiesKHR{};

    var features2 = types.PhysicalDeviceFeatures2{
        .p_next = &features,
    };

    var props2 = types.PhysicalDeviceProperties2{
        .s_type = .physical_device_properties_2,
        .p_next = &props,
    };

    if (ctx.instance_dispatch.vkGetPhysicalDeviceFeatures2) |getFeatures2| {
        getFeatures2(ctx.physical_device, &features2);
    }

    if (ctx.instance_dispatch.vkGetPhysicalDeviceProperties2) |getProps2| {
        getProps2(ctx.physical_device, &props2);
    }

    return RayTracingCapabilities{
        .ray_tracing_pipeline = features.ray_tracing_pipeline != 0,
        .trace_rays_indirect = features.ray_tracing_pipeline_trace_rays_indirect != 0,
        .ray_traversal_primitive_culling = features.ray_traversal_primitive_culling != 0,
        .max_ray_recursion_depth = props.max_ray_recursion_depth,
        .shader_group_handle_size = props.shader_group_handle_size,
        .max_ray_dispatch_invocation_count = props.max_ray_dispatch_invocation_count,
    };
}

// =============================================================================
// Video Queue
// =============================================================================

/// Video codec capabilities
pub const VideoCapabilities = struct {
    /// Supports H.264 encoding
    encode_h264: bool,
    /// Supports H.265/HEVC encoding
    encode_h265: bool,
    /// Supports H.264 decoding
    decode_h264: bool,
    /// Supports H.265/HEVC decoding
    decode_h265: bool,
    /// Supports AV1 decoding
    decode_av1: bool,
    /// Queue family index for video operations
    video_queue_family: ?u32,

    /// Check if any video encoding is supported
    pub fn supportsEncode(self: VideoCapabilities) bool {
        return self.encode_h264 or self.encode_h265;
    }

    /// Check if any video decoding is supported
    pub fn supportsDecode(self: VideoCapabilities) bool {
        return self.decode_h264 or self.decode_h265 or self.decode_av1;
    }

    /// Check if video queue is available
    pub fn isAvailable(self: VideoCapabilities) bool {
        return self.video_queue_family != null;
    }
};

/// Query video queue capabilities
///
/// **STUB IMPLEMENTATION**: This function currently returns a default "unavailable" result.
/// Full implementation requires:
/// - Querying VkQueueFamilyVideoPropertiesKHR for each queue family
/// - Checking for VK_KHR_video_queue extension support
/// - Querying VkVideoCapabilitiesKHR for codec-specific capabilities
///
/// Query video encoding/decoding capabilities.
///
/// NOTE: This is currently a stub implementation that returns all features disabled.
/// Actual implementation requires VK_KHR_video_queue extension which needs explicit
/// device enablement at Context.init() time. Callers should check individual capability
/// fields before using video features.
///
/// Returns: Default VideoCapabilities with all features disabled (stub)
pub fn queryVideoCapabilities(ctx: *Context, allocator: std.mem.Allocator) err.Error!VideoCapabilities {
    _ = ctx;
    _ = allocator;

    // STUB: Video capability detection not yet implemented
    // Requires: VK_KHR_video_queue, VK_KHR_video_encode_queue, VK_KHR_video_decode_queue
    // These must be enabled at device creation time
    std.log.debug("queryVideoCapabilities: stub - returning all features disabled", .{});

    return VideoCapabilities{
        .encode_h264 = false,
        .encode_h265 = false,
        .decode_h264 = false,
        .decode_h265 = false,
        .decode_av1 = false,
        .video_queue_family = null,
    };
}

// =============================================================================
// Cooperative Matrices
// =============================================================================

/// Common matrix configuration (e.g., for tensor cores)
pub const MatrixConfiguration = struct {
    m: u32,
    n: u32,
    k: u32,
    a_type: types.ComponentTypeKHR,
    b_type: types.ComponentTypeKHR,
    c_type: types.ComponentTypeKHR,
    result_type: types.ComponentTypeKHR,
    scope: types.ScopeKHR,
};

/// Cooperative matrix capabilities
pub const CooperativeMatrixCapabilities = struct {
    /// Cooperative matrix operations supported
    cooperative_matrix: bool,
    /// Robust buffer access for cooperative matrices
    robust_buffer_access: bool,
    /// Supported matrix configurations (caller-owned memory)
    configurations: []const MatrixConfiguration,

    /// Check if cooperative matrices are available
    pub fn isAvailable(self: CooperativeMatrixCapabilities) bool {
        return self.cooperative_matrix;
    }
};

/// Query cooperative matrix capabilities
/// Note: The allocator is used to store the configurations array. The caller owns this memory.
pub fn queryCooperativeMatrixCapabilities(ctx: *Context, allocator: std.mem.Allocator) err.Error!CooperativeMatrixCapabilities {
    var features = types.PhysicalDeviceCooperativeMatrixFeaturesKHR{};

    var features2 = types.PhysicalDeviceFeatures2{
        .p_next = &features,
    };

    if (ctx.instance_dispatch.vkGetPhysicalDeviceFeatures2) |getFeatures2| {
        getFeatures2(ctx.physical_device, &features2);
    } else {
        return err.Error.ExtensionNotPresent;
    }

    const is_supported = features.cooperative_matrix != 0;
    const robust_access = features.cooperative_matrix_robust_buffer_access != 0;

    // Query cooperative matrix configurations if the extension function is available
    var configurations: []MatrixConfiguration = &.{};

    if (ctx.instance_dispatch.vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) |getCoopMatrix| {
        // First call: get count
        var count: u32 = 0;
        const result1 = getCoopMatrix(ctx.physical_device, &count, null);
        if (result1 != .success or count == 0) {
            return CooperativeMatrixCapabilities{
                .cooperative_matrix = is_supported,
                .robust_buffer_access = robust_access,
                .configurations = configurations,
            };
        }

        // Allocate array for properties
        var props = allocator.alloc(types.CooperativeMatrixPropertiesKHR, count) catch {
            return CooperativeMatrixCapabilities{
                .cooperative_matrix = is_supported,
                .robust_buffer_access = robust_access,
                .configurations = configurations,
            };
        };
        defer allocator.free(props);

        // Initialize all with correct s_type
        for (props) |*p| {
            p.* = types.CooperativeMatrixPropertiesKHR{};
        }

        // Second call: get properties
        const result2 = getCoopMatrix(ctx.physical_device, &count, props.ptr);
        if (result2 == .success) {
            // Convert to our MatrixConfiguration format
            configurations = allocator.alloc(MatrixConfiguration, count) catch &.{};

            for (props[0..count], 0..) |prop, i| {
                configurations[i] = .{
                    .m = prop.m_size,
                    .n = prop.n_size,
                    .k = prop.k_size,
                    .a_type = prop.a_type,
                    .b_type = prop.b_type,
                    .c_type = prop.c_type,
                    .result_type = prop.result_type,
                    .scope = prop.scope,
                };
            }
        }
    }

    return CooperativeMatrixCapabilities{
        .cooperative_matrix = is_supported,
        .robust_buffer_access = robust_access,
        .configurations = configurations,
    };
}

// =============================================================================
// Extension Availability Check
// =============================================================================

/// All extension capabilities combined
pub const ExtensionCapabilities = struct {
    subgroup: ?SubgroupCapabilities,
    descriptor_indexing: ?DescriptorIndexingCapabilities,
    ray_tracing: ?RayTracingCapabilities,
    video: ?VideoCapabilities,
    cooperative_matrix: ?CooperativeMatrixCapabilities,
};

/// Query all extension capabilities at once
pub fn queryAllCapabilities(ctx: *Context, allocator: std.mem.Allocator) ExtensionCapabilities {
    return ExtensionCapabilities{
        .subgroup = querySubgroupProperties(ctx) catch null,
        .descriptor_indexing = queryDescriptorIndexingFeatures(ctx) catch null,
        .ray_tracing = queryRayTracingCapabilities(ctx) catch null,
        .video = queryVideoCapabilities(ctx, allocator) catch null,
        .cooperative_matrix = queryCooperativeMatrixCapabilities(ctx, allocator) catch null,
    };
}

/// Print extension capabilities summary
pub fn printCapabilitiesSummary(caps: ExtensionCapabilities, writer: anytype) !void {
    try writer.print("Extension Capabilities:\n", .{});
    try writer.print("========================\n", .{});

    if (caps.subgroup) |sg| {
        try writer.print("\nSubgroup Operations:\n", .{});
        try writer.print("  Size: {}\n", .{sg.subgroup_size});
        try writer.print("  Compute support: {}\n", .{sg.supportsCompute()});
        try writer.print("  Arithmetic: {}\n", .{sg.supportsOperation(.arithmetic)});
        try writer.print("  Ballot: {}\n", .{sg.supportsOperation(.ballot)});
        try writer.print("  Shuffle: {}\n", .{sg.supportsOperation(.shuffle)});
    } else {
        try writer.print("\nSubgroup Operations: Not available\n", .{});
    }

    if (caps.descriptor_indexing) |di| {
        try writer.print("\nDescriptor Indexing:\n", .{});
        try writer.print("  Bindless support: {}\n", .{di.supportsBindless()});
        try writer.print("  Runtime arrays: {}\n", .{di.runtime_descriptor_array});
        try writer.print("  Partially bound: {}\n", .{di.partially_bound});
    } else {
        try writer.print("\nDescriptor Indexing: Not available\n", .{});
    }

    if (caps.ray_tracing) |rt| {
        try writer.print("\nRay Tracing:\n", .{});
        try writer.print("  Available: {}\n", .{rt.isAvailable()});
        try writer.print("  Max recursion depth: {}\n", .{rt.max_ray_recursion_depth});
        try writer.print("  Indirect trace: {}\n", .{rt.trace_rays_indirect});
    } else {
        try writer.print("\nRay Tracing: Not available\n", .{});
    }

    if (caps.video) |vid| {
        try writer.print("\nVideo Queue:\n", .{});
        try writer.print("  Encode support: {}\n", .{vid.supportsEncode()});
        try writer.print("  Decode support: {}\n", .{vid.supportsDecode()});
    } else {
        try writer.print("\nVideo Queue: Not available\n", .{});
    }

    if (caps.cooperative_matrix) |cm| {
        try writer.print("\nCooperative Matrices:\n", .{});
        try writer.print("  Available: {}\n", .{cm.isAvailable()});
        try writer.print("  Robust access: {}\n", .{cm.robust_buffer_access});
        try writer.print("  Configurations: {}\n", .{cm.configurations.len});
        for (cm.configurations, 0..) |config, i| {
            try writer.print("    [{d}] M={d} N={d} K={d} ({s}x{s}->{s})\n", .{
                i,
                config.m,
                config.n,
                config.k,
                @tagName(config.a_type),
                @tagName(config.b_type),
                @tagName(config.result_type),
            });
        }
    } else {
        try writer.print("\nCooperative Matrices: Not available\n", .{});
    }
}

// =============================================================================
// Sparse Resources (VK_EXT_sparse_residency)
// =============================================================================

/// Sparse resource capabilities for virtual texturing
pub const SparseResourceCapabilities = struct {
    /// Sparse binding for buffers
    sparse_binding: bool,
    /// Sparse residency for buffers
    sparse_residency_buffer: bool,
    /// Sparse residency for 2D images
    sparse_residency_image_2d: bool,
    /// Sparse residency for 3D images
    sparse_residency_image_3d: bool,
    /// Sparse residency for 2-sample images
    sparse_residency_2_samples: bool,
    /// Sparse residency for 4-sample images
    sparse_residency_4_samples: bool,
    /// Sparse residency for 8-sample images
    sparse_residency_8_samples: bool,
    /// Sparse residency for 16-sample images
    sparse_residency_16_samples: bool,
    /// Sparse residency aliased
    sparse_residency_aliased: bool,
    /// Residency standard 2D block shape
    residency_standard_2d_block_shape: bool,
    /// Residency standard 3D block shape
    residency_standard_3d_block_shape: bool,
    /// Residency aligned mip size
    residency_aligned_mip_size: bool,
    /// Residency non-resident strict
    residency_non_resident_strict: bool,

    /// Check if sparse resources are available
    pub fn isAvailable(self: SparseResourceCapabilities) bool {
        return self.sparse_binding;
    }

    /// Check if virtual texturing is supported (2D sparse images)
    pub fn supportsVirtualTexturing(self: SparseResourceCapabilities) bool {
        return self.sparse_residency_image_2d and self.residency_standard_2d_block_shape;
    }

    /// Check if sparse buffers are supported
    pub fn supportsSparseBuffers(self: SparseResourceCapabilities) bool {
        return self.sparse_residency_buffer;
    }
};

/// Query sparse resource capabilities
pub fn querySparseResourceCapabilities(ctx: *Context) SparseResourceCapabilities {
    // Query device features for sparse support
    var features: types.PhysicalDeviceFeatures = undefined;
    ctx.instance_dispatch.vkGetPhysicalDeviceFeatures.?(ctx.physical_device, &features);

    // Query device properties for sparse properties
    var props: types.PhysicalDeviceProperties = undefined;
    ctx.instance_dispatch.vkGetPhysicalDeviceProperties.?(ctx.physical_device, &props);

    return SparseResourceCapabilities{
        .sparse_binding = features.sparse_binding != 0,
        .sparse_residency_buffer = features.sparse_residency_buffer != 0,
        .sparse_residency_image_2d = features.sparse_residency_image_2d != 0,
        .sparse_residency_image_3d = features.sparse_residency_image_3d != 0,
        .sparse_residency_2_samples = features.sparse_residency_2_samples != 0,
        .sparse_residency_4_samples = features.sparse_residency_4_samples != 0,
        .sparse_residency_8_samples = features.sparse_residency_8_samples != 0,
        .sparse_residency_16_samples = features.sparse_residency_16_samples != 0,
        .sparse_residency_aliased = features.sparse_residency_aliased != 0,
        .residency_standard_2d_block_shape = props.sparse_properties.residency_standard_2d_block_shape != 0,
        .residency_standard_3d_block_shape = props.sparse_properties.residency_standard_3d_block_shape != 0,
        .residency_aligned_mip_size = props.sparse_properties.residency_aligned_mip_size != 0,
        .residency_non_resident_strict = props.sparse_properties.residency_non_resident_strict != 0,
    };
}

// =============================================================================
// Mesh Shaders (VK_EXT_mesh_shader)
// =============================================================================

/// Mesh shader capabilities
pub const MeshShaderCapabilities = struct {
    /// Mesh shader is supported
    mesh_shader: bool,
    /// Task shader is supported
    task_shader: bool,
    /// Multiview mesh shader is supported
    multiview_mesh_shader: bool,
    /// Primitive fragment shading rate mesh shader
    primitive_fragment_shading_rate_mesh_shader: bool,
    /// Maximum output vertices
    max_mesh_output_vertices: u32,
    /// Maximum output primitives
    max_mesh_output_primitives: u32,
    /// Maximum mesh work group count
    max_mesh_work_group_count: [3]u32,
    /// Maximum task work group count
    max_task_work_group_count: [3]u32,
    /// Maximum mesh work group invocations
    max_mesh_work_group_invocations: u32,
    /// Maximum task work group invocations
    max_task_work_group_invocations: u32,
    /// Preferred mesh work group invocations
    preferred_mesh_work_group_invocations: u32,
    /// Preferred task work group invocations
    preferred_task_work_group_invocations: u32,

    /// Check if mesh shaders are available
    pub fn isAvailable(self: MeshShaderCapabilities) bool {
        return self.mesh_shader;
    }

    /// Check if task shaders are available (amplification shaders)
    pub fn hasTaskShaders(self: MeshShaderCapabilities) bool {
        return self.task_shader;
    }

    /// Get recommended work group size for mesh shaders
    pub fn recommendedMeshWorkGroupSize(self: MeshShaderCapabilities) u32 {
        return self.preferred_mesh_work_group_invocations;
    }
};

/// Query mesh shader capabilities
///
/// **STUB IMPLEMENTATION**: This function currently returns a default "unavailable" result.
/// Full implementation requires:
/// - Enabling VK_EXT_mesh_shader device extension at device creation
/// - Querying VkPhysicalDeviceMeshShaderFeaturesEXT
/// - Querying VkPhysicalDeviceMeshShaderPropertiesEXT
///
/// Note: Mesh shaders require explicit extension enablement during device creation.
/// Query mesh and task shader capabilities.
///
/// NOTE: This is currently a stub implementation that returns all features disabled.
/// Actual implementation requires VK_EXT_mesh_shader extension which needs explicit
/// device enablement at Context.init() time. Callers should check the mesh_shader
/// and task_shader fields before using these features.
///
/// Returns: Default MeshShaderCapabilities with all features disabled (stub)
pub fn queryMeshShaderCapabilities(ctx: *Context, allocator: std.mem.Allocator) MeshShaderCapabilities {
    _ = allocator;
    _ = ctx;

    // STUB: Mesh shader capability detection not yet implemented
    // Requires: VK_EXT_mesh_shader enabled at device creation time
    std.log.debug("queryMeshShaderCapabilities: stub - returning all features disabled", .{});

    return MeshShaderCapabilities{
        .mesh_shader = false,
        .task_shader = false,
        .multiview_mesh_shader = false,
        .primitive_fragment_shading_rate_mesh_shader = false,
        .max_mesh_output_vertices = 0,
        .max_mesh_output_primitives = 0,
        .max_mesh_work_group_count = .{ 0, 0, 0 },
        .max_task_work_group_count = .{ 0, 0, 0 },
        .max_mesh_work_group_invocations = 0,
        .max_task_work_group_invocations = 0,
        .preferred_mesh_work_group_invocations = 0,
        .preferred_task_work_group_invocations = 0,
    };
}

// =============================================================================
// Debug Printf (VK_KHR_shader_non_semantic_info)
// =============================================================================

/// Debug printf capabilities for shader debugging
pub const DebugPrintfCapabilities = struct {
    /// Non-semantic info is supported (enables debugPrintfEXT)
    non_semantic_info: bool,
    /// Maximum output buffer size for debug printf
    max_output_buffer_size: u32,

    /// Check if debug printf is available
    pub fn isAvailable(self: DebugPrintfCapabilities) bool {
        return self.non_semantic_info;
    }
};

/// Query debug printf capabilities
/// Note: Requires validation layers for actual printf output
pub fn queryDebugPrintfCapabilities(ctx: *Context) DebugPrintfCapabilities {
    // Check if validation layers are enabled and debug printf is available
    const has_debug = ctx.hasDebugUtils();

    return DebugPrintfCapabilities{
        .non_semantic_info = has_debug,
        .max_output_buffer_size = if (has_debug) 1024 * 1024 else 0, // 1MB default
    };
}

// =============================================================================
// Memory Budget (VK_EXT_memory_budget)
// =============================================================================

/// Memory heap budget information
pub const MemoryHeapBudget = struct {
    /// Total size of the heap in bytes
    heap_size: u64,
    /// Budget: how much the application can allocate
    budget: u64,
    /// Current usage in this heap
    usage: u64,

    /// Get available budget (budget - usage)
    pub fn available(self: MemoryHeapBudget) u64 {
        return if (self.budget > self.usage) self.budget - self.usage else 0;
    }

    /// Get usage percentage (0.0 - 1.0)
    pub fn usageRatio(self: MemoryHeapBudget) f64 {
        if (self.budget == 0) return 0.0;
        return @as(f64, @floatFromInt(self.usage)) / @as(f64, @floatFromInt(self.budget));
    }

    /// Check if budget is exceeded
    pub fn isOverBudget(self: MemoryHeapBudget) bool {
        return self.usage > self.budget;
    }
};

/// Complete memory budget information for all heaps
pub const MemoryBudgetInfo = struct {
    /// Budget info per heap (indexed by memory heap index)
    heaps: [types.VK_MAX_MEMORY_HEAPS]MemoryHeapBudget,
    /// Number of valid heap entries
    heap_count: u32,
    /// Whether memory budget extension is available
    budget_extension_available: bool,

    /// Get budget for a specific heap
    pub fn getHeap(self: MemoryBudgetInfo, heap_index: u32) ?MemoryHeapBudget {
        if (heap_index >= self.heap_count) return null;
        return self.heaps[heap_index];
    }

    /// Get total device-local memory budget
    pub fn getDeviceLocalBudget(self: MemoryBudgetInfo, memory_properties: types.PhysicalDeviceMemoryProperties) u64 {
        var total: u64 = 0;
        for (0..self.heap_count) |i| {
            const heap_flags = memory_properties.memory_heaps[i].flags;
            if (heap_flags.device_local) {
                total += self.heaps[i].budget;
            }
        }
        return total;
    }

    /// Get total device-local memory usage
    pub fn getDeviceLocalUsage(self: MemoryBudgetInfo, memory_properties: types.PhysicalDeviceMemoryProperties) u64 {
        var total: u64 = 0;
        for (0..self.heap_count) |i| {
            const heap_flags = memory_properties.memory_heaps[i].flags;
            if (heap_flags.device_local) {
                total += self.heaps[i].usage;
            }
        }
        return total;
    }

    /// Check if any heap is over budget
    pub fn isAnyHeapOverBudget(self: MemoryBudgetInfo) bool {
        for (0..self.heap_count) |i| {
            if (self.heaps[i].isOverBudget()) return true;
        }
        return false;
    }

    /// Get the heap with the lowest available budget ratio
    pub fn getMostConstrainedHeap(self: MemoryBudgetInfo) ?struct { index: u32, budget: MemoryHeapBudget } {
        if (self.heap_count == 0) return null;

        var min_ratio: f64 = 2.0; // > 1.0
        var min_idx: u32 = 0;

        for (0..self.heap_count) |i| {
            const ratio = 1.0 - self.heaps[i].usageRatio();
            if (ratio < min_ratio) {
                min_ratio = ratio;
                min_idx = @intCast(i);
            }
        }

        return .{ .index = min_idx, .budget = self.heaps[min_idx] };
    }
};

/// Query memory budget information
/// Returns real-time budget if VK_EXT_memory_budget is available,
/// otherwise returns heap sizes as budget with zero usage
pub fn queryMemoryBudget(ctx: *Context) MemoryBudgetInfo {
    const mem_props = ctx.memory_properties;
    var budget_info = MemoryBudgetInfo{
        .heaps = undefined,
        .heap_count = mem_props.memory_heap_count,
        .budget_extension_available = false,
    };

    // Initialize with default values (heap size = budget, no usage info)
    for (0..types.VK_MAX_MEMORY_HEAPS) |i| {
        if (i < mem_props.memory_heap_count) {
            budget_info.heaps[i] = .{
                .heap_size = mem_props.memory_heaps[i].size,
                .budget = mem_props.memory_heaps[i].size,
                .usage = 0,
            };
        } else {
            budget_info.heaps[i] = .{ .heap_size = 0, .budget = 0, .usage = 0 };
        }
    }

    // Try to get actual budget from VK_EXT_memory_budget
    if (ctx.instance_dispatch.vkGetPhysicalDeviceMemoryProperties2) |getMemProps2| {
        var budget_props = types.PhysicalDeviceMemoryBudgetPropertiesEXT{};
        var mem_props2 = types.PhysicalDeviceMemoryProperties2{
            .p_next = &budget_props,
        };

        getMemProps2(ctx.physical_device, &mem_props2);

        // Check if we got valid budget data (non-zero budgets indicate extension is working)
        var has_valid_budget = false;
        for (0..mem_props.memory_heap_count) |i| {
            if (budget_props.heap_budget[i] > 0) {
                has_valid_budget = true;
                break;
            }
        }

        if (has_valid_budget) {
            budget_info.budget_extension_available = true;
            for (0..mem_props.memory_heap_count) |i| {
                budget_info.heaps[i].budget = budget_props.heap_budget[i];
                budget_info.heaps[i].usage = budget_props.heap_usage[i];
            }
        }
    }

    return budget_info;
}

/// Callback for memory status change notifications.
///
/// ## Ownership
/// - The callback function pointer is borrowed; must remain valid while monitoring is active
/// - user_data is passed through unchanged; caller owns this memory
/// - The MemoryStatus struct is valid only for the callback duration; copy if needed
///
/// ## Thread Safety
/// Callbacks are invoked from the thread calling MemoryBudgetMonitor.poll().
pub const MemoryStatusCallback = *const fn (status: MemoryBudgetMonitor.MemoryStatus, user_data: ?*anyopaque) void;

/// Memory budget monitor for continuous monitoring
///
/// Provides both on-demand checking and periodic polling modes:
///
/// ## On-demand checking
/// ```zig
/// const monitor = MemoryBudgetMonitor.init(&ctx);
/// const status = monitor.check();
/// if (status.level == .critical) {
///     // Handle low memory
/// }
/// ```
///
/// ## Periodic polling (cooperative)
/// ```zig
/// var monitor = MemoryBudgetMonitor.initPolling(&ctx, .{
///     .poll_interval_ms = 100,
///     .callback = my_callback,
/// });
/// // In your event loop:
/// while (running) {
///     monitor.poll();  // Non-blocking, calls callback on status change
///     // ... other work ...
/// }
/// ```
///
/// ## Periodic polling (threaded)
/// ```zig
/// var monitor = MemoryBudgetMonitor.initPolling(&ctx, .{
///     .poll_interval_ms = 100,
///     .callback = my_callback,
/// });
/// try monitor.startPolling();  // Spawns background thread
/// // ... later ...
/// monitor.stopPolling();  // Stops background thread
/// ```
pub const MemoryBudgetMonitor = struct {
    ctx: *Context,
    /// Warning threshold (0.0 - 1.0), emit warning when usage exceeds this
    warning_threshold: f64,
    /// Critical threshold (0.0 - 1.0), emit critical when usage exceeds this
    critical_threshold: f64,

    // Polling state (protected by mutex when background thread is running)
    poll_interval_ms: u64 = 100,
    callback: ?MemoryStatusCallback = null,
    callback_user_data: ?*anyopaque = null,
    last_status: ?MemoryStatus = null,
    last_poll_time: std.atomic.Value(i128) = std.atomic.Value(i128).init(0),
    polling_thread: ?std.Thread = null,
    stop_polling: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    // Thread-safe statistics using atomics
    total_checks: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    status_changes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    // Mutex for protecting callback and last_status modifications
    mutex: std.Thread.Mutex = .{},

    const Self = @This();

    /// Polling configuration
    pub const PollingConfig = struct {
        /// Polling interval in milliseconds
        poll_interval_ms: u64 = 100,
        /// Callback for status changes
        callback: ?MemoryStatusCallback = null,
        /// User data passed to callback
        user_data: ?*anyopaque = null,
        /// Warning threshold (0.0 - 1.0)
        warning_threshold: f64 = 0.8,
        /// Critical threshold (0.0 - 1.0)
        critical_threshold: f64 = 0.95,
        /// Only notify on status level changes (not every poll)
        notify_on_change_only: bool = true,
    };

    /// Create a new budget monitor with default thresholds
    pub fn init(ctx: *Context) Self {
        return .{
            .ctx = ctx,
            .warning_threshold = 0.8, // 80% usage
            .critical_threshold = 0.95, // 95% usage
        };
    }

    /// Create with custom thresholds
    pub fn initWithThresholds(ctx: *Context, warning: f64, critical: f64) Self {
        return .{
            .ctx = ctx,
            .warning_threshold = warning,
            .critical_threshold = critical,
        };
    }

    /// Create a polling monitor with configuration
    pub fn initPolling(ctx: *Context, config: PollingConfig) Self {
        return .{
            .ctx = ctx,
            .warning_threshold = config.warning_threshold,
            .critical_threshold = config.critical_threshold,
            .poll_interval_ms = config.poll_interval_ms,
            .callback = config.callback,
            .callback_user_data = config.user_data,
        };
    }

    /// Check current memory status (thread-safe)
    pub fn check(self: *Self) MemoryStatus {
        const budget = queryMemoryBudget(self.ctx);

        var max_usage_ratio: f64 = 0.0;
        var critical_heaps: u32 = 0;
        var warning_heaps: u32 = 0;

        for (0..budget.heap_count) |i| {
            const ratio = budget.heaps[i].usageRatio();
            if (ratio > max_usage_ratio) max_usage_ratio = ratio;

            if (ratio >= self.critical_threshold) {
                critical_heaps += 1;
            } else if (ratio >= self.warning_threshold) {
                warning_heaps += 1;
            }
        }

        _ = self.total_checks.fetchAdd(1, .monotonic);

        return MemoryStatus{
            .budget = budget,
            .max_usage_ratio = max_usage_ratio,
            .critical_heap_count = critical_heaps,
            .warning_heap_count = warning_heaps,
            .level = if (critical_heaps > 0)
                .critical
            else if (warning_heaps > 0)
                .warning
            else
                .ok,
        };
    }

    /// Cooperative polling - call from your event loop (thread-safe)
    ///
    /// This is a non-blocking check that:
    /// 1. Returns immediately if poll interval hasn't elapsed
    /// 2. Checks memory status if interval has elapsed
    /// 3. Calls callback if status changed (or always, depending on config)
    ///
    /// Returns true if a check was performed, false if skipped
    pub fn poll(self: *Self) bool {
        const now = std.time.nanoTimestamp();
        const last_time = self.last_poll_time.load(.acquire);
        const elapsed_ns = now - last_time;
        const interval_ns = @as(i128, self.poll_interval_ms) * 1_000_000;

        if (elapsed_ns < interval_ns) {
            return false;
        }

        // Try to update last_poll_time atomically to avoid multiple threads polling
        if (self.last_poll_time.cmpxchgStrong(last_time, now, .acq_rel, .acquire) != null) {
            // Another thread already updated, skip this poll
            return false;
        }

        const status = self.check();

        // Lock for callback and last_status access
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check if status level changed
        const changed = if (self.last_status) |last| last.level != status.level else true;
        if (changed) {
            _ = self.status_changes.fetchAdd(1, .monotonic);
        }

        // Call callback while holding lock (callback should be quick)
        if (self.callback) |cb| {
            if (changed or self.last_status == null) {
                cb(status, self.callback_user_data);
            }
        }

        self.last_status = status;
        return true;
    }

    /// Start background polling thread
    ///
    /// The thread will poll at the configured interval and call the callback
    /// when status changes. Use stopPolling() to stop the thread.
    pub fn startPolling(self: *Self) !void {
        if (self.polling_thread != null) {
            return; // Already polling
        }

        self.stop_polling.store(false, .release);
        self.polling_thread = try std.Thread.spawn(.{}, pollingThreadFn, .{self});
    }

    /// Stop background polling thread
    pub fn stopPolling(self: *Self) void {
        if (self.polling_thread) |thread| {
            self.stop_polling.store(true, .release);
            thread.join();
            self.polling_thread = null;
        }
    }

    fn pollingThreadFn(self: *Self) void {
        while (!self.stop_polling.load(.acquire)) {
            _ = self.poll();
            std.time.sleep(self.poll_interval_ms * 1_000_000);
        }
    }

    /// Set the callback for status changes (thread-safe)
    pub fn setCallback(self: *Self, callback: ?MemoryStatusCallback, user_data: ?*anyopaque) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.callback = callback;
        self.callback_user_data = user_data;
    }

    /// Get monitoring statistics (thread-safe)
    pub fn getStats(self: *Self) MonitorStats {
        self.mutex.lock();
        const last_status = self.last_status;
        const is_polling = self.polling_thread != null;
        self.mutex.unlock();

        return .{
            .total_checks = self.total_checks.load(.acquire),
            .status_changes = self.status_changes.load(.acquire),
            .last_status = last_status,
            .is_polling = is_polling,
        };
    }

    /// Monitoring statistics
    pub const MonitorStats = struct {
        total_checks: u64,
        status_changes: u64,
        last_status: ?MemoryStatus,
        is_polling: bool,
    };

    /// Memory status levels
    pub const Level = enum {
        ok,
        warning,
        critical,
    };

    /// Memory status result
    pub const MemoryStatus = struct {
        budget: MemoryBudgetInfo,
        max_usage_ratio: f64,
        critical_heap_count: u32,
        warning_heap_count: u32,
        level: Level,

        /// Check if memory pressure is acceptable
        pub fn isOk(self: MemoryStatus) bool {
            return self.level == .ok;
        }

        /// Format status as string
        pub fn format(self: MemoryStatus, writer: anytype) !void {
            const level_str = switch (self.level) {
                .ok => "OK",
                .warning => "WARNING",
                .critical => "CRITICAL",
            };
            try writer.print("Memory Status: {s} (max usage: {d:.1}%)\n", .{
                level_str,
                self.max_usage_ratio * 100,
            });

            for (0..self.budget.heap_count) |i| {
                const heap = self.budget.heaps[i];
                try writer.print("  Heap {}: {d:.1} MB / {d:.1} MB ({d:.1}%)\n", .{
                    i,
                    @as(f64, @floatFromInt(heap.usage)) / (1024 * 1024),
                    @as(f64, @floatFromInt(heap.budget)) / (1024 * 1024),
                    heap.usageRatio() * 100,
                });
            }
        }
    };
};

// =============================================================================
// Memory Pressure Handler (Automatic Eviction)
// =============================================================================

/// Eviction callback type for memory-constrained situations.
///
/// ## Ownership
/// - The callback function pointer is borrowed; must remain valid until policy is removed
/// - user_data is passed through unchanged; caller owns this memory
///
/// ## Thread Safety
/// Callbacks are invoked from the thread attempting the allocation that
/// triggered eviction. Ensure your callback is thread-safe if allocations
/// can occur from multiple threads.
///
/// ## Expected Behavior
/// The callback should attempt to free at least `bytes_needed` bytes.
/// Return the actual number of bytes freed in the EvictionResult.
pub const EvictionCallback = *const fn (user_data: ?*anyopaque, bytes_needed: usize) EvictionResult;

/// Result of an eviction attempt
pub const EvictionResult = struct {
    /// Bytes actually freed
    bytes_freed: usize,
    /// Whether eviction was successful
    success: bool,
};

/// Evictable resource registration
pub const EvictableResource = struct {
    /// User data passed to callback
    user_data: ?*anyopaque,
    /// Eviction callback
    evict_fn: EvictionCallback,
    /// Human-readable name for debugging
    name: []const u8,
    /// Priority (lower = evicted first)
    priority: i32,
    /// Estimated size in bytes
    size_bytes: usize,
    /// Last access timestamp (nanoseconds)
    last_access: i128,
    /// Whether resource is currently in use (don't evict)
    in_use: bool,
};

/// Memory pressure handler configuration
pub const MemoryPressureConfig = struct {
    /// Warning threshold (0.0 - 1.0)
    warning_threshold: f64 = 0.75,
    /// Critical threshold (0.0 - 1.0)
    critical_threshold: f64 = 0.90,
    /// Target threshold after eviction (0.0 - 1.0)
    target_threshold: f64 = 0.70,
    /// Maximum resources to track
    max_resources: usize = 256,
    /// Check interval in milliseconds (for periodic checks)
    check_interval_ms: u64 = 100,
    /// Enable automatic eviction
    auto_evict: bool = true,
    /// Log eviction events
    enable_logging: bool = true,
};

/// Memory pressure handler with automatic eviction
///
/// Manages memory pressure by:
/// 1. Monitoring memory usage against configurable thresholds
/// 2. Tracking evictable resources by priority
/// 3. Automatically evicting low-priority resources when needed
///
/// ## Usage
///
/// ```zig
/// var pressure = MemoryPressureHandler.init(allocator, ctx, .{});
/// defer pressure.deinit();
///
/// // Register evictable resources
/// try pressure.registerResource("texture_cache", cache_ptr, evictCallback, 10, cache_size);
///
/// // Check pressure before allocations
/// if (pressure.checkAndEvict(required_bytes)) {
///     // Safe to allocate
/// } else {
///     // Not enough memory even after eviction
/// }
/// ```
pub const MemoryPressureHandler = struct {
    allocator: std.mem.Allocator,
    ctx: *Context,
    config: MemoryPressureConfig,
    resources: std.ArrayListUnmanaged(EvictableResource),
    stats: PressureStats,
    monitor: MemoryBudgetMonitor,

    const Self = @This();

    /// Pressure handling statistics
    pub const PressureStats = struct {
        /// Total bytes evicted
        total_bytes_evicted: usize = 0,
        /// Number of eviction events
        eviction_count: u32 = 0,
        /// Number of resources currently tracked
        tracked_resources: usize = 0,
        /// Last check time
        last_check_time: i128 = 0,
        /// Current pressure level
        current_level: MemoryBudgetMonitor.Level = .ok,
    };

    /// Initialize memory pressure handler
    pub fn init(allocator: std.mem.Allocator, ctx: *Context, config: MemoryPressureConfig) Self {
        return Self{
            .allocator = allocator,
            .ctx = ctx,
            .config = config,
            .resources = .{},
            .stats = .{},
            .monitor = MemoryBudgetMonitor.initWithThresholds(ctx, config.warning_threshold, config.critical_threshold),
        };
    }

    /// Deinitialize
    pub fn deinit(self: *Self) void {
        self.resources.deinit(self.allocator);
    }

    /// Register an evictable resource
    ///
    /// Returns error.TooManyObjects if max_resources limit is reached.
    pub fn registerResource(
        self: *Self,
        name: []const u8,
        user_data: ?*anyopaque,
        evict_fn: EvictionCallback,
        priority: i32,
        size_bytes: usize,
    ) err.Error!void {
        if (self.resources.items.len >= self.config.max_resources) {
            std.log.warn("MemoryPressureHandler.registerResource: resource limit reached ({d}/{d})", .{
                self.resources.items.len,
                self.config.max_resources,
            });
            return err.Error.TooManyObjects;
        }

        try self.resources.append(self.allocator, .{
            .user_data = user_data,
            .evict_fn = evict_fn,
            .name = name,
            .priority = priority,
            .size_bytes = size_bytes,
            .last_access = std.time.nanoTimestamp(),
            .in_use = false,
        });

        self.stats.tracked_resources = self.resources.items.len;
    }

    /// Unregister a resource
    pub fn unregisterResource(self: *Self, user_data: ?*anyopaque) void {
        var i: usize = 0;
        while (i < self.resources.items.len) {
            if (self.resources.items[i].user_data == user_data) {
                _ = self.resources.orderedRemove(i);
            } else {
                i += 1;
            }
        }
        self.stats.tracked_resources = self.resources.items.len;
    }

    /// Mark a resource as in-use (prevents eviction)
    pub fn markInUse(self: *Self, user_data: ?*anyopaque, in_use: bool) void {
        for (self.resources.items) |*r| {
            if (r.user_data == user_data) {
                r.in_use = in_use;
                if (!in_use) {
                    r.last_access = std.time.nanoTimestamp();
                }
                return;
            }
        }
    }

    /// Update resource size estimate
    pub fn updateSize(self: *Self, user_data: ?*anyopaque, new_size: usize) void {
        for (self.resources.items) |*r| {
            if (r.user_data == user_data) {
                r.size_bytes = new_size;
                return;
            }
        }
    }

    /// Check current memory pressure level
    pub fn check(self: *Self) MemoryBudgetMonitor.MemoryStatus {
        const status = self.monitor.check();
        self.stats.last_check_time = std.time.nanoTimestamp();
        self.stats.current_level = status.level;
        return status;
    }

    /// Check pressure and automatically evict if needed
    /// Returns true if memory is within acceptable limits (after eviction if needed)
    pub fn checkAndEvict(self: *Self, bytes_needed: usize) bool {
        const status = self.check();

        if (status.level == .ok) {
            return true;
        }

        if (!self.config.auto_evict) {
            return status.level != .critical;
        }

        // Calculate how much we need to free
        const budget_info = status.budget;
        var bytes_to_free: usize = bytes_needed;

        // Add extra to get below target threshold
        if (budget_info.budget_extension_available) {
            const total_budget = budget_info.getDeviceLocalBudget(self.ctx.memory_properties);
            const total_usage = budget_info.getDeviceLocalUsage(self.ctx.memory_properties);
            // Calculate target usage with bounds checking to prevent overflow
            const target_f = @as(f64, @floatFromInt(total_budget)) * self.config.target_threshold;
            const target_capped = @min(target_f, @as(f64, std.math.maxInt(usize)));
            const target_usage: usize = if (target_capped < 0) 0 else @intFromFloat(target_capped);

            if (total_usage > target_usage) {
                bytes_to_free = @max(bytes_to_free, total_usage - target_usage);
            }
        }

        // Attempt eviction
        const freed = self.evict(bytes_to_free);
        return freed >= bytes_needed;
    }

    /// Evict resources to free at least bytes_needed
    pub fn evict(self: *Self, bytes_needed: usize) usize {
        // Sort resources by eviction priority (lower priority first, then by LRU)
        std.mem.sort(EvictableResource, self.resources.items, {}, struct {
            fn lessThan(_: void, a: EvictableResource, b: EvictableResource) bool {
                if (a.in_use != b.in_use) return !a.in_use; // Non-in-use first
                if (a.priority != b.priority) return a.priority < b.priority;
                return a.last_access < b.last_access; // Older first
            }
        }.lessThan);

        var total_freed: usize = 0;
        var evicted_indices = std.ArrayListUnmanaged(usize){};
        defer evicted_indices.deinit(self.allocator);

        for (self.resources.items, 0..) |*resource, idx| {
            if (total_freed >= bytes_needed) break;
            if (resource.in_use) continue;

            if (self.config.enable_logging) {
                std.log.info("Memory pressure: evicting '{s}' ({d} bytes)", .{ resource.name, resource.size_bytes });
            }

            const result = resource.evict_fn(resource.user_data, bytes_needed - total_freed);
            if (result.success) {
                total_freed += result.bytes_freed;
                self.stats.total_bytes_evicted += result.bytes_freed;
                self.stats.eviction_count += 1;
                evicted_indices.append(self.allocator, idx) catch |e| {
                    // If we can't track evicted index, resource remains in list but is already freed
                    std.log.warn("MemoryPressureHandler.evict: failed to track evicted resource index: {s} - resource list may be stale", .{
                        @errorName(e),
                    });
                };
            }
        }

        // Remove evicted resources (in reverse order to maintain indices)
        var i: usize = evicted_indices.items.len;
        while (i > 0) {
            i -= 1;
            _ = self.resources.orderedRemove(evicted_indices.items[i]);
        }

        self.stats.tracked_resources = self.resources.items.len;
        return total_freed;
    }

    /// Get current statistics
    pub fn getStats(self: Self) PressureStats {
        return self.stats;
    }

    /// Get total tracked resource size
    pub fn getTotalTrackedSize(self: Self) usize {
        var total: usize = 0;
        for (self.resources.items) |r| {
            total += r.size_bytes;
        }
        return total;
    }

    /// Check if any resources can be evicted
    pub fn hasEvictableResources(self: Self) bool {
        for (self.resources.items) |r| {
            if (!r.in_use) return true;
        }
        return false;
    }
};

// =============================================================================
// Push Descriptors (VK_KHR_push_descriptor)
// =============================================================================

/// Push descriptor capabilities
pub const PushDescriptorCapabilities = struct {
    /// Push descriptors are supported
    push_descriptor: bool,
    /// Maximum number of push descriptors
    max_push_descriptors: u32,

    /// Check if push descriptors are available
    pub fn isAvailable(self: PushDescriptorCapabilities) bool {
        return self.push_descriptor and self.max_push_descriptors > 0;
    }
};

/// Query push descriptor capabilities
pub fn queryPushDescriptorCapabilities(ctx: *Context) PushDescriptorCapabilities {
    var props = types.PhysicalDevicePushDescriptorPropertiesKHR{};
    var props2 = types.PhysicalDeviceProperties2{
        .s_type = .physical_device_properties_2,
        .p_next = &props,
    };

    if (ctx.instance_dispatch.vkGetPhysicalDeviceProperties2) |getProps2| {
        getProps2(ctx.physical_device, &props2);

        return PushDescriptorCapabilities{
            .push_descriptor = props.max_push_descriptors > 0,
            .max_push_descriptors = props.max_push_descriptors,
        };
    }

    return PushDescriptorCapabilities{
        .push_descriptor = false,
        .max_push_descriptors = 0,
    };
}

// =============================================================================
// Extended capabilities query
// =============================================================================

/// Extended extension capabilities (includes sparse, mesh, debug, memory budget, push descriptors)
pub const ExtendedCapabilities = struct {
    /// Sparse resource capabilities
    sparse: ?SparseResourceCapabilities,
    /// Mesh shader capabilities
    mesh_shader: ?MeshShaderCapabilities,
    /// Debug printf capabilities
    debug_printf: ?DebugPrintfCapabilities,
    /// Memory budget capabilities
    memory_budget: ?MemoryBudgetInfo,
    /// Push descriptor capabilities
    push_descriptor: ?PushDescriptorCapabilities,
};

/// Query all extended capabilities
pub fn queryExtendedCapabilities(ctx: *Context, allocator: std.mem.Allocator) ExtendedCapabilities {
    return ExtendedCapabilities{
        .sparse = querySparseResourceCapabilities(ctx),
        .mesh_shader = queryMeshShaderCapabilities(ctx, allocator),
        .debug_printf = queryDebugPrintfCapabilities(ctx),
        .memory_budget = queryMemoryBudget(ctx),
        .push_descriptor = queryPushDescriptorCapabilities(ctx),
    };
}

// =============================================================================
// Tests
// =============================================================================

test "SparseResourceCapabilities" {
    const caps = SparseResourceCapabilities{
        .sparse_binding = true,
        .sparse_residency_buffer = true,
        .sparse_residency_image_2d = true,
        .sparse_residency_image_3d = false,
        .sparse_residency_2_samples = false,
        .sparse_residency_4_samples = false,
        .sparse_residency_8_samples = false,
        .sparse_residency_16_samples = false,
        .sparse_residency_aliased = false,
        .residency_standard_2d_block_shape = true,
        .residency_standard_3d_block_shape = false,
        .residency_aligned_mip_size = false,
        .residency_non_resident_strict = false,
    };
    try std.testing.expect(caps.isAvailable());
    try std.testing.expect(caps.supportsVirtualTexturing());
    try std.testing.expect(caps.supportsSparseBuffers());
}

test "MeshShaderCapabilities" {
    const caps = MeshShaderCapabilities{
        .mesh_shader = true,
        .task_shader = true,
        .multiview_mesh_shader = false,
        .primitive_fragment_shading_rate_mesh_shader = false,
        .max_mesh_output_vertices = 256,
        .max_mesh_output_primitives = 512,
        .max_mesh_work_group_count = .{ 65535, 65535, 65535 },
        .max_task_work_group_count = .{ 65535, 65535, 65535 },
        .max_mesh_work_group_invocations = 128,
        .max_task_work_group_invocations = 128,
        .preferred_mesh_work_group_invocations = 32,
        .preferred_task_work_group_invocations = 32,
    };
    try std.testing.expect(caps.isAvailable());
    try std.testing.expect(caps.hasTaskShaders());
    try std.testing.expectEqual(@as(u32, 32), caps.recommendedMeshWorkGroupSize());
}

test "DebugPrintfCapabilities" {
    const caps = DebugPrintfCapabilities{
        .non_semantic_info = true,
        .max_output_buffer_size = 1024 * 1024,
    };
    try std.testing.expect(caps.isAvailable());
}

test "SubgroupCapabilities operations check" {
    const caps = SubgroupCapabilities{
        .subgroup_size = 32,
        .supported_stages = .{ .compute = true },
        .supported_operations = .{ .basic = true, .arithmetic = true, .ballot = true },
        .quad_operations_in_all_stages = false,
    };

    try std.testing.expect(caps.supportsOperation(.basic));
    try std.testing.expect(caps.supportsOperation(.arithmetic));
    try std.testing.expect(caps.supportsOperation(.ballot));
    try std.testing.expect(!caps.supportsOperation(.shuffle));
    try std.testing.expect(caps.supportsCompute());
}

test "DescriptorIndexingCapabilities bindless check" {
    const caps_full = DescriptorIndexingCapabilities{
        .runtime_descriptor_array = true,
        .storage_buffer_update_after_bind = true,
        .sampled_image_update_after_bind = true,
        .partially_bound = true,
        .variable_descriptor_count = true,
        .storage_buffer_non_uniform_indexing = true,
        .sampled_image_non_uniform_indexing = true,
    };
    try std.testing.expect(caps_full.supportsBindless());

    const caps_partial = DescriptorIndexingCapabilities{
        .runtime_descriptor_array = true,
        .storage_buffer_update_after_bind = false,
        .sampled_image_update_after_bind = false,
        .partially_bound = false,
        .variable_descriptor_count = true,
        .storage_buffer_non_uniform_indexing = false,
        .sampled_image_non_uniform_indexing = false,
    };
    try std.testing.expect(!caps_partial.supportsBindless());
}

test "RayTracingCapabilities availability" {
    const caps = RayTracingCapabilities{
        .ray_tracing_pipeline = true,
        .trace_rays_indirect = true,
        .ray_traversal_primitive_culling = false,
        .max_ray_recursion_depth = 31,
        .shader_group_handle_size = 32,
        .max_ray_dispatch_invocation_count = 1073741824,
    };
    try std.testing.expect(caps.isAvailable());
}

test "VideoCapabilities encode/decode" {
    const caps = VideoCapabilities{
        .encode_h264 = true,
        .encode_h265 = false,
        .decode_h264 = true,
        .decode_h265 = true,
        .decode_av1 = false,
        .video_queue_family = 2,
    };
    try std.testing.expect(caps.supportsEncode());
    try std.testing.expect(caps.supportsDecode());
    try std.testing.expect(caps.isAvailable());
}

test "MemoryHeapBudget calculations" {
    const heap = MemoryHeapBudget{
        .heap_size = 8 * 1024 * 1024 * 1024, // 8 GB
        .budget = 6 * 1024 * 1024 * 1024, // 6 GB budget
        .usage = 2 * 1024 * 1024 * 1024, // 2 GB used
    };

    // Available should be 4 GB
    try std.testing.expectEqual(@as(u64, 4 * 1024 * 1024 * 1024), heap.available());

    // Usage ratio should be ~0.33
    const ratio = heap.usageRatio();
    try std.testing.expect(ratio > 0.32 and ratio < 0.34);

    // Not over budget
    try std.testing.expect(!heap.isOverBudget());

    // Over budget case
    const over_budget = MemoryHeapBudget{
        .heap_size = 4 * 1024 * 1024 * 1024,
        .budget = 2 * 1024 * 1024 * 1024,
        .usage = 3 * 1024 * 1024 * 1024,
    };
    try std.testing.expect(over_budget.isOverBudget());
    try std.testing.expectEqual(@as(u64, 0), over_budget.available());
}

test "PushDescriptorCapabilities" {
    const caps = PushDescriptorCapabilities{
        .push_descriptor = true,
        .max_push_descriptors = 32,
    };
    try std.testing.expect(caps.isAvailable());

    const no_push = PushDescriptorCapabilities{
        .push_descriptor = false,
        .max_push_descriptors = 0,
    };
    try std.testing.expect(!no_push.isAvailable());
}
