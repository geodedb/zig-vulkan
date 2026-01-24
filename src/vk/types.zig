// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Vulkan API Types and Constants
//!
//! This module contains all Vulkan type definitions, enumerations, and constants
//! needed for compute operations. Types are defined to match the Vulkan specification
//! with Zig-idiomatic naming where appropriate.

const std = @import("std");

// =============================================================================
// Vulkan Handle Types
// =============================================================================

pub const Instance = *opaque {};
pub const PhysicalDevice = *opaque {};
pub const Device = *opaque {};
pub const Queue = *opaque {};
pub const CommandPool = *opaque {};
pub const CommandBuffer = *opaque {};
pub const Buffer = *opaque {};
pub const Image = *opaque {};
pub const ImageView = *opaque {};
pub const DeviceMemory = *opaque {};
pub const ShaderModule = *opaque {};
pub const DescriptorSetLayout = *opaque {};
pub const DescriptorPool = *opaque {};
pub const DescriptorSet = *opaque {};
pub const PipelineLayout = *opaque {};
pub const PipelineCache = *opaque {};
pub const Pipeline = *opaque {};
pub const Fence = *opaque {};
pub const Semaphore = *opaque {};
pub const QueryPool = *opaque {};
pub const Sampler = *opaque {};
pub const BufferView = *opaque {};
pub const DescriptorUpdateTemplate = *opaque {};
pub const Event = *opaque {};

/// Null handle value for optional handles
pub const NULL_HANDLE = @as(?*anyopaque, null);

// =============================================================================
// Vulkan Constants
// =============================================================================

pub const MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;
pub const UUID_SIZE: usize = 16;
pub const MAX_MEMORY_TYPES: usize = 32;
pub const MAX_MEMORY_HEAPS: usize = 16;
pub const VK_MAX_MEMORY_HEAPS: usize = 16; // Alias for Vulkan spec naming
pub const MAX_EXTENSION_NAME_SIZE: usize = 256;
pub const MAX_DESCRIPTION_SIZE: usize = 256;
pub const QUEUE_FAMILY_IGNORED: u32 = 0xFFFFFFFF;
pub const WHOLE_SIZE: u64 = 0xFFFFFFFFFFFFFFFF;

// API Version helpers
pub const API_VERSION_1_0: u32 = makeApiVersion(0, 1, 0, 0);
pub const API_VERSION_1_1: u32 = makeApiVersion(0, 1, 1, 0);
pub const API_VERSION_1_2: u32 = makeApiVersion(0, 1, 2, 0);
pub const API_VERSION_1_3: u32 = makeApiVersion(0, 1, 3, 0);

pub fn makeApiVersion(variant: u32, major: u32, minor: u32, patch: u32) u32 {
    return (variant << 29) | (major << 22) | (minor << 12) | patch;
}

pub fn apiVersionMajor(version: u32) u32 {
    return (version >> 22) & 0x7F;
}

pub fn apiVersionMinor(version: u32) u32 {
    return (version >> 12) & 0x3FF;
}

pub fn apiVersionPatch(version: u32) u32 {
    return version & 0xFFF;
}

// =============================================================================
// Vulkan Result Codes
// =============================================================================

pub const Result = enum(i32) {
    success = 0,
    not_ready = 1,
    timeout = 2,
    event_set = 3,
    event_reset = 4,
    incomplete = 5,
    error_out_of_host_memory = -1,
    error_out_of_device_memory = -2,
    error_initialization_failed = -3,
    error_device_lost = -4,
    error_memory_map_failed = -5,
    error_layer_not_present = -6,
    error_extension_not_present = -7,
    error_feature_not_present = -8,
    error_incompatible_driver = -9,
    error_too_many_objects = -10,
    error_format_not_supported = -11,
    error_fragmented_pool = -12,
    error_unknown = -13,
    error_out_of_pool_memory = -1000069000,
    error_invalid_external_handle = -1000072003,
    error_fragmentation = -1000161000,
    error_invalid_opaque_capture_address = -1000257000,
    error_surface_lost_khr = -1000000000,
    error_native_window_in_use_khr = -1000000001,
    suboptimal_khr = 1000001003,
    error_out_of_date_khr = -1000001004,
    _,

    pub fn isSuccess(self: Result) bool {
        return @intFromEnum(self) >= 0;
    }

    pub fn isError(self: Result) bool {
        return @intFromEnum(self) < 0;
    }
};

// =============================================================================
// Structure Types
// =============================================================================

pub const StructureType = enum(i32) {
    application_info = 0,
    instance_create_info = 1,
    device_queue_create_info = 2,
    device_create_info = 3,
    submit_info = 4,
    memory_allocate_info = 5,
    mapped_memory_range = 6,
    bind_sparse_info = 7,
    fence_create_info = 8,
    semaphore_create_info = 9,
    event_create_info = 10,
    query_pool_create_info = 11,
    buffer_create_info = 12,
    buffer_view_create_info = 13,
    image_create_info = 14,
    image_view_create_info = 15,
    shader_module_create_info = 16,
    pipeline_cache_create_info = 17,
    pipeline_shader_stage_create_info = 18,
    compute_pipeline_create_info = 29,
    pipeline_layout_create_info = 30,
    sampler_create_info = 31,
    descriptor_set_layout_create_info = 32,
    descriptor_pool_create_info = 33,
    descriptor_set_allocate_info = 34,
    write_descriptor_set = 35,
    copy_descriptor_set = 36,
    command_pool_create_info = 39,
    command_buffer_allocate_info = 40,
    command_buffer_inheritance_info = 41,
    command_buffer_begin_info = 42,
    memory_barrier = 46,
    buffer_memory_barrier = 44,
    image_memory_barrier = 45,
    physical_device_features_2 = 1000059000,
    physical_device_properties_2 = 1000059001,
    debug_utils_messenger_create_info_ext = 1000128004,
    debug_utils_object_name_info_ext = 1000128000,
    debug_utils_label_ext = 1000128002,
    debug_utils_messenger_callback_data_ext = 1000128003,
    submit_info_2 = 1000314004,
    // Timeline semaphores (Vulkan 1.2)
    semaphore_type_create_info = 1000207002,
    timeline_semaphore_submit_info = 1000207003,
    semaphore_wait_info = 1000207004,
    semaphore_signal_info = 1000207005,
    // Subgroup operations (Vulkan 1.1)
    physical_device_subgroup_properties = 1000094000,
    // Descriptor indexing (VK_EXT_descriptor_indexing / Vulkan 1.2)
    descriptor_set_layout_binding_flags_create_info = 1000161000,
    physical_device_descriptor_indexing_features = 1000161001,
    physical_device_descriptor_indexing_properties = 1000161002,
    descriptor_set_variable_descriptor_count_allocate_info = 1000161003,
    descriptor_set_variable_descriptor_count_layout_support = 1000161004,
    // Ray tracing (VK_KHR_ray_tracing_pipeline)
    physical_device_ray_tracing_pipeline_features_khr = 1000347000,
    physical_device_ray_tracing_pipeline_properties_khr = 1000347001,
    ray_tracing_pipeline_create_info_khr = 1000150015,
    ray_tracing_shader_group_create_info_khr = 1000150016,
    ray_tracing_pipeline_interface_create_info_khr = 1000150018,
    // Acceleration structure (VK_KHR_acceleration_structure)
    acceleration_structure_create_info_khr = 1000150017,
    acceleration_structure_build_geometry_info_khr = 1000150000,
    acceleration_structure_geometry_khr = 1000150006,
    acceleration_structure_build_sizes_info_khr = 1000150020,
    physical_device_acceleration_structure_features_khr = 1000150013,
    physical_device_acceleration_structure_properties_khr = 1000150014,
    // Video (VK_KHR_video_queue)
    video_session_create_info_khr = 1000023001,
    video_session_parameters_create_info_khr = 1000023002,
    video_begin_coding_info_khr = 1000023004,
    video_end_coding_info_khr = 1000023005,
    video_coding_control_info_khr = 1000023006,
    physical_device_video_format_info_khr = 1000023014,
    video_format_properties_khr = 1000023015,
    queue_family_video_properties_khr = 1000023012,
    // Cooperative matrices (VK_KHR_cooperative_matrix)
    physical_device_cooperative_matrix_features_khr = 1000506000,
    physical_device_cooperative_matrix_properties_khr = 1000506001,
    cooperative_matrix_properties_khr = 1000506002,
    // Memory budget (VK_EXT_memory_budget)
    physical_device_memory_budget_properties_ext = 1000237000,
    physical_device_memory_properties_2 = 1000059006,
    // Push descriptors (VK_KHR_push_descriptor)
    physical_device_push_descriptor_properties_khr = 1000080000,
    // Descriptor update templates (Vulkan 1.1)
    descriptor_update_template_create_info = 1000085000,
    // Debug utils object tag
    debug_utils_object_tag_info_ext = 1000128001,
    _,
};

// =============================================================================
// Queue and Device Types
// =============================================================================

pub const QueueFlags = packed struct(u32) {
    graphics: bool = false,
    compute: bool = false,
    transfer: bool = false,
    sparse_binding: bool = false,
    protected: bool = false,
    _padding: u27 = 0,
};

pub const QueueFamilyProperties = extern struct {
    queue_flags: QueueFlags,
    queue_count: u32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: Extent3D,
};

pub const PhysicalDeviceType = enum(i32) {
    other = 0,
    integrated_gpu = 1,
    discrete_gpu = 2,
    virtual_gpu = 3,
    cpu = 4,
    _,
};

pub const PhysicalDeviceLimits = extern struct {
    max_image_dimension_1d: u32,
    max_image_dimension_2d: u32,
    max_image_dimension_3d: u32,
    max_image_dimension_cube: u32,
    max_image_array_layers: u32,
    max_texel_buffer_elements: u32,
    max_uniform_buffer_range: u32,
    max_storage_buffer_range: u32,
    max_push_constants_size: u32,
    max_memory_allocation_count: u32,
    max_sampler_allocation_count: u32,
    buffer_image_granularity: u64,
    sparse_address_space_size: u64,
    max_bound_descriptor_sets: u32,
    max_per_stage_descriptor_samplers: u32,
    max_per_stage_descriptor_uniform_buffers: u32,
    max_per_stage_descriptor_storage_buffers: u32,
    max_per_stage_descriptor_sampled_images: u32,
    max_per_stage_descriptor_storage_images: u32,
    max_per_stage_descriptor_input_attachments: u32,
    max_per_stage_resources: u32,
    max_descriptor_set_samplers: u32,
    max_descriptor_set_uniform_buffers: u32,
    max_descriptor_set_uniform_buffers_dynamic: u32,
    max_descriptor_set_storage_buffers: u32,
    max_descriptor_set_storage_buffers_dynamic: u32,
    max_descriptor_set_sampled_images: u32,
    max_descriptor_set_storage_images: u32,
    max_descriptor_set_input_attachments: u32,
    max_vertex_input_attributes: u32,
    max_vertex_input_bindings: u32,
    max_vertex_input_attribute_offset: u32,
    max_vertex_input_binding_stride: u32,
    max_vertex_output_components: u32,
    max_tessellation_generation_level: u32,
    max_tessellation_patch_size: u32,
    max_tessellation_control_per_vertex_input_components: u32,
    max_tessellation_control_per_vertex_output_components: u32,
    max_tessellation_control_per_patch_output_components: u32,
    max_tessellation_control_total_output_components: u32,
    max_tessellation_evaluation_input_components: u32,
    max_tessellation_evaluation_output_components: u32,
    max_geometry_shader_invocations: u32,
    max_geometry_input_components: u32,
    max_geometry_output_components: u32,
    max_geometry_output_vertices: u32,
    max_geometry_total_output_components: u32,
    max_fragment_input_components: u32,
    max_fragment_output_attachments: u32,
    max_fragment_dual_src_attachments: u32,
    max_fragment_combined_output_resources: u32,
    max_compute_shared_memory_size: u32,
    max_compute_work_group_count: [3]u32,
    max_compute_work_group_invocations: u32,
    max_compute_work_group_size: [3]u32,
    sub_pixel_precision_bits: u32,
    sub_texel_precision_bits: u32,
    mipmap_precision_bits: u32,
    max_draw_indexed_index_value: u32,
    max_draw_indirect_count: u32,
    max_sampler_lod_bias: f32,
    max_sampler_anisotropy: f32,
    max_viewports: u32,
    max_viewport_dimensions: [2]u32,
    viewport_bounds_range: [2]f32,
    viewport_sub_pixel_bits: u32,
    min_memory_map_alignment: usize,
    min_texel_buffer_offset_alignment: u64,
    min_uniform_buffer_offset_alignment: u64,
    min_storage_buffer_offset_alignment: u64,
    min_texel_offset: i32,
    max_texel_offset: u32,
    min_texel_gather_offset: i32,
    max_texel_gather_offset: u32,
    min_interpolation_offset: f32,
    max_interpolation_offset: f32,
    sub_pixel_interpolation_offset_bits: u32,
    max_framebuffer_width: u32,
    max_framebuffer_height: u32,
    max_framebuffer_layers: u32,
    framebuffer_color_sample_counts: u32,
    framebuffer_depth_sample_counts: u32,
    framebuffer_stencil_sample_counts: u32,
    framebuffer_no_attachments_sample_counts: u32,
    max_color_attachments: u32,
    sampled_image_color_sample_counts: u32,
    sampled_image_integer_sample_counts: u32,
    sampled_image_depth_sample_counts: u32,
    sampled_image_stencil_sample_counts: u32,
    storage_image_sample_counts: u32,
    max_sample_mask_words: u32,
    timestamp_compute_and_graphics: u32,
    timestamp_period: f32,
    max_clip_distances: u32,
    max_cull_distances: u32,
    max_combined_clip_and_cull_distances: u32,
    discrete_queue_priorities: u32,
    point_size_range: [2]f32,
    line_width_range: [2]f32,
    point_size_granularity: f32,
    line_width_granularity: f32,
    strict_lines: u32,
    standard_sample_locations: u32,
    optimal_buffer_copy_offset_alignment: u64,
    optimal_buffer_copy_row_pitch_alignment: u64,
    non_coherent_atom_size: u64,
};

pub const PhysicalDeviceSparseProperties = extern struct {
    residency_standard_2d_block_shape: u32,
    residency_standard_2d_multisample_block_shape: u32,
    residency_standard_3d_block_shape: u32,
    residency_aligned_mip_size: u32,
    residency_non_resident_strict: u32,
};

pub const PhysicalDeviceProperties = extern struct {
    api_version: u32,
    driver_version: u32,
    vendor_id: u32,
    device_id: u32,
    device_type: PhysicalDeviceType,
    device_name: [MAX_PHYSICAL_DEVICE_NAME_SIZE]u8,
    pipeline_cache_uuid: [UUID_SIZE]u8,
    limits: PhysicalDeviceLimits,
    sparse_properties: PhysicalDeviceSparseProperties,

    pub fn getDeviceName(self: *const PhysicalDeviceProperties) []const u8 {
        const name = &self.device_name;
        const len = std.mem.indexOfScalar(u8, name, 0) orelse name.len;
        return name[0..len];
    }
};

/// Physical device properties with extension chain (Vulkan 1.1+)
pub const PhysicalDeviceProperties2 = extern struct {
    s_type: StructureType = .physical_device_properties_2,
    p_next: ?*anyopaque = null,
    properties: PhysicalDeviceProperties = undefined,
};

/// Physical device features with extension chain (Vulkan 1.1+)
/// Note: PhysicalDeviceFeatures is defined later in this file
pub const PhysicalDeviceFeatures2 = extern struct {
    s_type: StructureType = .physical_device_features_2,
    p_next: ?*anyopaque = null,
    features: PhysicalDeviceFeatures = .{},
};

// =============================================================================
// Memory Types
// =============================================================================

pub const MemoryPropertyFlags = packed struct(u32) {
    device_local: bool = false,
    host_visible: bool = false,
    host_coherent: bool = false,
    host_cached: bool = false,
    lazily_allocated: bool = false,
    protected: bool = false,
    _padding: u26 = 0,
};

pub const MemoryHeapFlags = packed struct(u32) {
    device_local: bool = false,
    multi_instance: bool = false,
    _padding: u30 = 0,
};

pub const MemoryType = extern struct {
    property_flags: MemoryPropertyFlags,
    heap_index: u32,
};

pub const MemoryHeap = extern struct {
    size: u64,
    flags: MemoryHeapFlags,
};

pub const PhysicalDeviceMemoryProperties = extern struct {
    memory_type_count: u32,
    memory_types: [MAX_MEMORY_TYPES]MemoryType,
    memory_heap_count: u32,
    memory_heaps: [MAX_MEMORY_HEAPS]MemoryHeap,
};

pub const MemoryAllocateInfo = extern struct {
    s_type: StructureType = .memory_allocate_info,
    p_next: ?*const anyopaque = null,
    allocation_size: u64,
    memory_type_index: u32,
};

pub const MappedMemoryRange = extern struct {
    s_type: StructureType = .mapped_memory_range,
    p_next: ?*const anyopaque = null,
    memory: DeviceMemory,
    offset: u64,
    size: u64,
};

// =============================================================================
// Buffer Types
// =============================================================================

pub const BufferUsageFlags = packed struct(u32) {
    transfer_src: bool = false,
    transfer_dst: bool = false,
    uniform_texel_buffer: bool = false,
    storage_texel_buffer: bool = false,
    uniform_buffer: bool = false,
    storage_buffer: bool = false,
    index_buffer: bool = false,
    vertex_buffer: bool = false,
    indirect_buffer: bool = false,
    _padding: u23 = 0,
};

pub const SharingMode = enum(i32) {
    exclusive = 0,
    concurrent = 1,
};

pub const BufferCreateInfo = extern struct {
    s_type: StructureType = .buffer_create_info,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    size: u64,
    usage: BufferUsageFlags,
    sharing_mode: SharingMode = .exclusive,
    queue_family_index_count: u32 = 0,
    p_queue_family_indices: ?[*]const u32 = null,
};

pub const MemoryRequirements = extern struct {
    size: u64,
    alignment: u64,
    memory_type_bits: u32,
};

// =============================================================================
// Command Buffer Types
// =============================================================================

pub const CommandPoolCreateFlags = packed struct(u32) {
    transient: bool = false,
    reset_command_buffer: bool = false,
    protected: bool = false,
    _padding: u29 = 0,
};

pub const CommandPoolCreateInfo = extern struct {
    s_type: StructureType = .command_pool_create_info,
    p_next: ?*const anyopaque = null,
    flags: CommandPoolCreateFlags = .{},
    queue_family_index: u32,
};

pub const CommandPoolResetFlags = packed struct(u32) {
    release_resources: bool = false,
    _padding: u31 = 0,
};

pub const CommandBufferResetFlags = packed struct(u32) {
    release_resources: bool = false,
    _padding: u31 = 0,
};

pub const CommandBufferLevel = enum(i32) {
    primary = 0,
    secondary = 1,
};

pub const CommandBufferAllocateInfo = extern struct {
    s_type: StructureType = .command_buffer_allocate_info,
    p_next: ?*const anyopaque = null,
    command_pool: CommandPool,
    level: CommandBufferLevel = .primary,
    command_buffer_count: u32,
};

pub const CommandBufferUsageFlags = packed struct(u32) {
    one_time_submit: bool = false,
    render_pass_continue: bool = false,
    simultaneous_use: bool = false,
    _padding: u29 = 0,
};

pub const CommandBufferBeginInfo = extern struct {
    s_type: StructureType = .command_buffer_begin_info,
    p_next: ?*const anyopaque = null,
    flags: CommandBufferUsageFlags = .{},
    p_inheritance_info: ?*const anyopaque = null,
};

// =============================================================================
// Pipeline Types
// =============================================================================

pub const ShaderModuleCreateInfo = extern struct {
    s_type: StructureType = .shader_module_create_info,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    code_size: usize,
    p_code: [*]const u32,
};

pub const ShaderStageFlags = packed struct(u32) {
    vertex: bool = false,
    tessellation_control: bool = false,
    tessellation_evaluation: bool = false,
    geometry: bool = false,
    fragment: bool = false,
    compute: bool = false,
    all_graphics: bool = false,
    all: bool = false,
    _padding: u24 = 0,
};

pub const PipelineShaderStageCreateInfo = extern struct {
    s_type: StructureType = .pipeline_shader_stage_create_info,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: ShaderStageFlags,
    module: ShaderModule,
    p_name: [*:0]const u8,
    p_specialization_info: ?*const anyopaque = null,
};

pub const PipelineLayoutCreateInfo = extern struct {
    s_type: StructureType = .pipeline_layout_create_info,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    set_layout_count: u32 = 0,
    p_set_layouts: ?[*]const DescriptorSetLayout = null,
    push_constant_range_count: u32 = 0,
    p_push_constant_ranges: ?[*]const PushConstantRange = null,
};

pub const PushConstantRange = extern struct {
    stage_flags: ShaderStageFlags,
    offset: u32,
    size: u32,
};

pub const ComputePipelineCreateInfo = extern struct {
    s_type: StructureType = .compute_pipeline_create_info,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: PipelineShaderStageCreateInfo,
    layout: PipelineLayout,
    base_pipeline_handle: ?Pipeline = null,
    base_pipeline_index: i32 = -1,
};

// =============================================================================
// Descriptor Types
// =============================================================================

pub const DescriptorType = enum(i32) {
    sampler = 0,
    combined_image_sampler = 1,
    sampled_image = 2,
    storage_image = 3,
    uniform_texel_buffer = 4,
    storage_texel_buffer = 5,
    uniform_buffer = 6,
    storage_buffer = 7,
    uniform_buffer_dynamic = 8,
    storage_buffer_dynamic = 9,
    input_attachment = 10,
    _,
};

pub const DescriptorSetLayoutBinding = extern struct {
    binding: u32,
    descriptor_type: DescriptorType,
    descriptor_count: u32,
    stage_flags: ShaderStageFlags,
    p_immutable_samplers: ?*const anyopaque = null,
};

pub const DescriptorSetLayoutCreateInfo = extern struct {
    s_type: StructureType = .descriptor_set_layout_create_info,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    binding_count: u32,
    p_bindings: [*]const DescriptorSetLayoutBinding,
};

pub const DescriptorPoolSize = extern struct {
    type: DescriptorType,
    descriptor_count: u32,
};

pub const DescriptorPoolCreateFlags = packed struct(u32) {
    free_descriptor_set: bool = false,
    update_after_bind: bool = false,
    _padding: u30 = 0,
};

pub const DescriptorPoolCreateInfo = extern struct {
    s_type: StructureType = .descriptor_pool_create_info,
    p_next: ?*const anyopaque = null,
    flags: DescriptorPoolCreateFlags = .{},
    max_sets: u32,
    pool_size_count: u32,
    p_pool_sizes: [*]const DescriptorPoolSize,
};

pub const DescriptorSetAllocateInfo = extern struct {
    s_type: StructureType = .descriptor_set_allocate_info,
    p_next: ?*const anyopaque = null,
    descriptor_pool: DescriptorPool,
    descriptor_set_count: u32,
    p_set_layouts: [*]const DescriptorSetLayout,
};

pub const DescriptorBufferInfo = extern struct {
    buffer: Buffer,
    offset: u64,
    range: u64,
};

pub const DescriptorImageInfo = extern struct {
    sampler: ?Sampler = null,
    image_view: ?ImageView = null,
    image_layout: ImageLayout = .undefined,
};

pub const WriteDescriptorSet = extern struct {
    s_type: StructureType = .write_descriptor_set,
    p_next: ?*const anyopaque = null,
    dst_set: ?DescriptorSet = null, // Optional for push descriptors (ignored per spec)
    dst_binding: u32,
    dst_array_element: u32 = 0,
    descriptor_count: u32,
    descriptor_type: DescriptorType,
    p_image_info: ?*const DescriptorImageInfo = null,
    p_buffer_info: ?*const DescriptorBufferInfo = null,
    p_texel_buffer_view: ?*const BufferView = null,
};

pub const CopyDescriptorSet = extern struct {
    s_type: StructureType = .copy_descriptor_set,
    p_next: ?*const anyopaque = null,
    src_set: DescriptorSet,
    src_binding: u32,
    src_array_element: u32 = 0,
    dst_set: DescriptorSet,
    dst_binding: u32,
    dst_array_element: u32 = 0,
    descriptor_count: u32,
};

pub const DescriptorUpdateTemplateType = enum(i32) {
    descriptor_set = 0,
    push_descriptors_khr = 1,
    _,
};

pub const DescriptorUpdateTemplateEntry = extern struct {
    dst_binding: u32,
    dst_array_element: u32 = 0,
    descriptor_count: u32,
    descriptor_type: DescriptorType,
    offset: usize,
    stride: usize,
};

pub const DescriptorUpdateTemplateCreateInfo = extern struct {
    s_type: StructureType = .descriptor_update_template_create_info,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    descriptor_update_entry_count: u32,
    p_descriptor_update_entries: [*]const DescriptorUpdateTemplateEntry,
    template_type: DescriptorUpdateTemplateType = .descriptor_set,
    descriptor_set_layout: ?DescriptorSetLayout = null,
    pipeline_bind_point: PipelineBindPoint = .compute,
    pipeline_layout: ?PipelineLayout = null,
    set: u32 = 0,
};

// =============================================================================
// Synchronization Types
// =============================================================================

pub const FenceCreateFlags = packed struct(u32) {
    signaled: bool = false,
    _padding: u31 = 0,
};

pub const FenceCreateInfo = extern struct {
    s_type: StructureType = .fence_create_info,
    p_next: ?*const anyopaque = null,
    flags: FenceCreateFlags = .{},
};

pub const SemaphoreCreateInfo = extern struct {
    s_type: StructureType = .semaphore_create_info,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
};

pub const EventCreateInfo = extern struct {
    s_type: StructureType = .event_create_info,
    p_next: ?*const anyopaque = null,
    flags: EventCreateFlags = .{},
};

pub const EventCreateFlags = packed struct(u32) {
    /// VK_EVENT_CREATE_DEVICE_ONLY_BIT - Event is only signaled/waited on the device
    device_only: bool = false,
    _padding: u31 = 0,
};

pub const PipelineStageFlags = packed struct(u32) {
    top_of_pipe: bool = false,
    draw_indirect: bool = false,
    vertex_input: bool = false,
    vertex_shader: bool = false,
    tessellation_control_shader: bool = false,
    tessellation_evaluation_shader: bool = false,
    geometry_shader: bool = false,
    fragment_shader: bool = false,
    early_fragment_tests: bool = false,
    late_fragment_tests: bool = false,
    color_attachment_output: bool = false,
    compute_shader: bool = false,
    transfer: bool = false,
    bottom_of_pipe: bool = false,
    host: bool = false,
    all_graphics: bool = false,
    all_commands: bool = false,
    _padding: u15 = 0,
};

pub const AccessFlags = packed struct(u32) {
    indirect_command_read: bool = false,
    index_read: bool = false,
    vertex_attribute_read: bool = false,
    uniform_read: bool = false,
    input_attachment_read: bool = false,
    shader_read: bool = false,
    shader_write: bool = false,
    color_attachment_read: bool = false,
    color_attachment_write: bool = false,
    depth_stencil_attachment_read: bool = false,
    depth_stencil_attachment_write: bool = false,
    transfer_read: bool = false,
    transfer_write: bool = false,
    host_read: bool = false,
    host_write: bool = false,
    memory_read: bool = false,
    memory_write: bool = false,
    _padding: u15 = 0,
};

pub const MemoryBarrier = extern struct {
    s_type: StructureType = .memory_barrier,
    p_next: ?*const anyopaque = null,
    src_access_mask: AccessFlags,
    dst_access_mask: AccessFlags,
};

pub const BufferMemoryBarrier = extern struct {
    s_type: StructureType = .buffer_memory_barrier,
    p_next: ?*const anyopaque = null,
    src_access_mask: AccessFlags,
    dst_access_mask: AccessFlags,
    src_queue_family_index: u32 = QUEUE_FAMILY_IGNORED,
    dst_queue_family_index: u32 = QUEUE_FAMILY_IGNORED,
    buffer: Buffer,
    offset: u64,
    size: u64,
};

pub const ImageMemoryBarrier = extern struct {
    s_type: StructureType = .image_memory_barrier,
    p_next: ?*const anyopaque = null,
    src_access_mask: AccessFlags,
    dst_access_mask: AccessFlags,
    old_layout: ImageLayout,
    new_layout: ImageLayout,
    src_queue_family_index: u32 = QUEUE_FAMILY_IGNORED,
    dst_queue_family_index: u32 = QUEUE_FAMILY_IGNORED,
    image: Image,
    subresource_range: ImageSubresourceRange,
};

pub const ImageLayout = enum(i32) {
    undefined = 0,
    general = 1,
    color_attachment_optimal = 2,
    depth_stencil_attachment_optimal = 3,
    depth_stencil_read_only_optimal = 4,
    shader_read_only_optimal = 5,
    transfer_src_optimal = 6,
    transfer_dst_optimal = 7,
    preinitialized = 8,
    _,
};

pub const ImageSubresourceRange = extern struct {
    aspect_mask: ImageAspectFlags,
    base_mip_level: u32,
    level_count: u32,
    base_array_layer: u32,
    layer_count: u32,
};

pub const ImageAspectFlags = packed struct(u32) {
    color: bool = false,
    depth: bool = false,
    stencil: bool = false,
    metadata: bool = false,
    _padding: u28 = 0,
};

pub const DependencyFlags = packed struct(u32) {
    by_region: bool = false,
    device_group: bool = false,
    view_local: bool = false,
    _padding: u29 = 0,
};

pub const PipelineBindPoint = enum(i32) {
    graphics = 0,
    compute = 1,
    ray_tracing_khr = 1000165000,
    _,
};

// =============================================================================
// Submit Types
// =============================================================================

pub const SubmitInfo = extern struct {
    s_type: StructureType = .submit_info,
    p_next: ?*const anyopaque = null,
    wait_semaphore_count: u32 = 0,
    p_wait_semaphores: ?[*]const Semaphore = null,
    p_wait_dst_stage_mask: ?[*]const PipelineStageFlags = null,
    command_buffer_count: u32,
    p_command_buffers: [*]const CommandBuffer,
    signal_semaphore_count: u32 = 0,
    p_signal_semaphores: ?[*]const Semaphore = null,
};

// =============================================================================
// Query Types
// =============================================================================

pub const QueryType = enum(i32) {
    occlusion = 0,
    pipeline_statistics = 1,
    timestamp = 2,
    _,
};

pub const QueryPoolCreateInfo = extern struct {
    s_type: StructureType = .query_pool_create_info,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    query_type: QueryType,
    query_count: u32,
    pipeline_statistics: u32 = 0,
};

pub const QueryResultFlags = packed struct(u32) {
    result_64: bool = false,
    wait: bool = false,
    with_availability: bool = false,
    partial: bool = false,
    _padding: u28 = 0,
};

// =============================================================================
// Instance/Device Creation Types
// =============================================================================

pub const ApplicationInfo = extern struct {
    s_type: StructureType = .application_info,
    p_next: ?*const anyopaque = null,
    p_application_name: ?[*:0]const u8 = null,
    application_version: u32 = 0,
    p_engine_name: ?[*:0]const u8 = null,
    engine_version: u32 = 0,
    api_version: u32 = API_VERSION_1_0,
};

pub const InstanceCreateInfo = extern struct {
    s_type: StructureType = .instance_create_info,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    p_application_info: ?*const ApplicationInfo = null,
    enabled_layer_count: u32 = 0,
    pp_enabled_layer_names: ?[*]const [*:0]const u8 = null,
    enabled_extension_count: u32 = 0,
    pp_enabled_extension_names: ?[*]const [*:0]const u8 = null,
};

pub const DeviceQueueCreateInfo = extern struct {
    s_type: StructureType = .device_queue_create_info,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    queue_family_index: u32,
    queue_count: u32,
    p_queue_priorities: [*]const f32,
};

pub const PhysicalDeviceFeatures = extern struct {
    robust_buffer_access: u32 = 0,
    full_draw_index_uint32: u32 = 0,
    image_cube_array: u32 = 0,
    independent_blend: u32 = 0,
    geometry_shader: u32 = 0,
    tessellation_shader: u32 = 0,
    sample_rate_shading: u32 = 0,
    dual_src_blend: u32 = 0,
    logic_op: u32 = 0,
    multi_draw_indirect: u32 = 0,
    draw_indirect_first_instance: u32 = 0,
    depth_clamp: u32 = 0,
    depth_bias_clamp: u32 = 0,
    fill_mode_non_solid: u32 = 0,
    depth_bounds: u32 = 0,
    wide_lines: u32 = 0,
    large_points: u32 = 0,
    alpha_to_one: u32 = 0,
    multi_viewport: u32 = 0,
    sampler_anisotropy: u32 = 0,
    texture_compression_etc2: u32 = 0,
    texture_compression_astc_ldr: u32 = 0,
    texture_compression_bc: u32 = 0,
    occlusion_query_precise: u32 = 0,
    pipeline_statistics_query: u32 = 0,
    vertex_pipeline_stores_and_atomics: u32 = 0,
    fragment_stores_and_atomics: u32 = 0,
    shader_tessellation_and_geometry_point_size: u32 = 0,
    shader_image_gather_extended: u32 = 0,
    shader_storage_image_extended_formats: u32 = 0,
    shader_storage_image_multisample: u32 = 0,
    shader_storage_image_read_without_format: u32 = 0,
    shader_storage_image_write_without_format: u32 = 0,
    shader_uniform_buffer_array_dynamic_indexing: u32 = 0,
    shader_sampled_image_array_dynamic_indexing: u32 = 0,
    shader_storage_buffer_array_dynamic_indexing: u32 = 0,
    shader_storage_image_array_dynamic_indexing: u32 = 0,
    shader_clip_distance: u32 = 0,
    shader_cull_distance: u32 = 0,
    shader_float64: u32 = 0,
    shader_int64: u32 = 0,
    shader_int16: u32 = 0,
    shader_resource_residency: u32 = 0,
    shader_resource_min_lod: u32 = 0,
    sparse_binding: u32 = 0,
    sparse_residency_buffer: u32 = 0,
    sparse_residency_image_2d: u32 = 0,
    sparse_residency_image_3d: u32 = 0,
    sparse_residency_2_samples: u32 = 0,
    sparse_residency_4_samples: u32 = 0,
    sparse_residency_8_samples: u32 = 0,
    sparse_residency_16_samples: u32 = 0,
    sparse_residency_aliased: u32 = 0,
    variable_multisample_rate: u32 = 0,
    inherited_queries: u32 = 0,
};

pub const DeviceCreateInfo = extern struct {
    s_type: StructureType = .device_create_info,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    queue_create_info_count: u32,
    p_queue_create_infos: [*]const DeviceQueueCreateInfo,
    enabled_layer_count: u32 = 0,
    pp_enabled_layer_names: ?[*]const [*:0]const u8 = null,
    enabled_extension_count: u32 = 0,
    pp_enabled_extension_names: ?[*]const [*:0]const u8 = null,
    p_enabled_features: ?*const PhysicalDeviceFeatures = null,
};

// =============================================================================
// Miscellaneous Types
// =============================================================================

pub const Extent3D = extern struct {
    width: u32,
    height: u32,
    depth: u32,
};

pub const BufferCopy = extern struct {
    src_offset: u64,
    dst_offset: u64,
    size: u64,
};

// =============================================================================
// Indirect Command Structures
// =============================================================================

/// Indirect dispatch command structure (VkDispatchIndirectCommand)
/// Used with vkCmdDispatchIndirect for GPU-driven workload dispatch
pub const DispatchIndirectCommand = extern struct {
    /// Number of workgroups in X dimension
    x: u32,
    /// Number of workgroups in Y dimension
    y: u32,
    /// Number of workgroups in Z dimension
    z: u32,
};

// =============================================================================
// Specialization Constants
// =============================================================================

/// Specialization map entry (VkSpecializationMapEntry)
/// Maps a constant ID to data in the specialization data block
pub const SpecializationMapEntry = extern struct {
    /// The ID of the specialization constant in SPIR-V
    constant_id: u32,
    /// Byte offset into the specialization data block
    offset: u32,
    /// Size in bytes of the specialization constant
    size: usize,
};

/// Specialization info for shader module creation (VkSpecializationInfo)
/// Provides specialization constant values at pipeline creation time
pub const SpecializationInfo = extern struct {
    /// Number of specialization map entries
    map_entry_count: u32 = 0,
    /// Pointer to array of SpecializationMapEntry
    p_map_entries: ?[*]const SpecializationMapEntry = null,
    /// Size in bytes of the specialization data
    data_size: usize = 0,
    /// Pointer to the specialization data block
    p_data: ?*const anyopaque = null,
};

// =============================================================================
// Timeline Semaphores (Vulkan 1.2+)
// =============================================================================

/// Semaphore type (VkSemaphoreType)
pub const SemaphoreType = enum(u32) {
    binary = 0,
    timeline = 1,
};

/// Semaphore type create info (VkSemaphoreTypeCreateInfo)
pub const SemaphoreTypeCreateInfo = extern struct {
    s_type: StructureType = .semaphore_type_create_info,
    p_next: ?*const anyopaque = null,
    semaphore_type: SemaphoreType = .binary,
    initial_value: u64 = 0,
};

/// Timeline semaphore submit info (VkTimelineSemaphoreSubmitInfo)
pub const TimelineSemaphoreSubmitInfo = extern struct {
    s_type: StructureType = .timeline_semaphore_submit_info,
    p_next: ?*const anyopaque = null,
    wait_semaphore_value_count: u32 = 0,
    p_wait_semaphore_values: ?[*]const u64 = null,
    signal_semaphore_value_count: u32 = 0,
    p_signal_semaphore_values: ?[*]const u64 = null,
};

/// Semaphore wait info (VkSemaphoreWaitInfo)
pub const SemaphoreWaitInfo = extern struct {
    s_type: StructureType = .semaphore_wait_info,
    p_next: ?*const anyopaque = null,
    flags: SemaphoreWaitFlags = .{},
    semaphore_count: u32 = 0,
    p_semaphores: ?[*]const Semaphore = null,
    p_values: ?[*]const u64 = null,
};

/// Semaphore signal info (VkSemaphoreSignalInfo)
pub const SemaphoreSignalInfo = extern struct {
    s_type: StructureType = .semaphore_signal_info,
    p_next: ?*const anyopaque = null,
    semaphore: Semaphore,
    value: u64,
};

/// Semaphore wait flags (VkSemaphoreWaitFlags)
pub const SemaphoreWaitFlags = packed struct(u32) {
    any: bool = false,
    _padding: u31 = 0,
};

// =============================================================================
// Extension Properties
// =============================================================================

pub const ExtensionProperties = extern struct {
    extension_name: [MAX_EXTENSION_NAME_SIZE]u8,
    spec_version: u32,

    pub fn getName(self: *const ExtensionProperties) []const u8 {
        const name = &self.extension_name;
        const len = std.mem.indexOfScalar(u8, name, 0) orelse name.len;
        return name[0..len];
    }
};

pub const LayerProperties = extern struct {
    layer_name: [MAX_EXTENSION_NAME_SIZE]u8,
    spec_version: u32,
    implementation_version: u32,
    description: [MAX_DESCRIPTION_SIZE]u8,

    pub fn getName(self: *const LayerProperties) []const u8 {
        const name = &self.layer_name;
        const len = std.mem.indexOfScalar(u8, name, 0) orelse name.len;
        return name[0..len];
    }
};

// =============================================================================
// Debug Utils (VK_EXT_debug_utils)
// =============================================================================

pub const DebugUtilsMessageSeverityFlagsEXT = packed struct(u32) {
    verbose: bool = false,
    _padding1: u3 = 0,
    info: bool = false,
    _padding2: u3 = 0,
    warning: bool = false,
    _padding3: u3 = 0,
    err: bool = false,
    _padding4: u19 = 0,
};

pub const DebugUtilsMessageTypeFlagsEXT = packed struct(u32) {
    general: bool = false,
    validation: bool = false,
    performance: bool = false,
    _padding: u29 = 0,
};

pub const DebugUtilsMessengerCallbackDataEXT = extern struct {
    s_type: StructureType = .debug_utils_messenger_callback_data_ext,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    p_message_id_name: ?[*:0]const u8 = null,
    message_id_number: i32 = 0,
    p_message: ?[*:0]const u8 = null,
    queue_label_count: u32 = 0,
    p_queue_labels: ?*const anyopaque = null,
    cmd_buf_label_count: u32 = 0,
    p_cmd_buf_labels: ?*const anyopaque = null,
    object_count: u32 = 0,
    p_objects: ?*const anyopaque = null,
};

pub const DebugUtilsMessengerCallbackFn = *const fn (
    DebugUtilsMessageSeverityFlagsEXT,
    DebugUtilsMessageTypeFlagsEXT,
    *const DebugUtilsMessengerCallbackDataEXT,
    ?*anyopaque,
) callconv(.c) u32;

pub const DebugUtilsMessengerCreateInfoEXT = extern struct {
    s_type: StructureType = .debug_utils_messenger_create_info_ext,
    p_next: ?*const anyopaque = null,
    flags: u32 = 0,
    message_severity: DebugUtilsMessageSeverityFlagsEXT,
    message_type: DebugUtilsMessageTypeFlagsEXT,
    pfn_user_callback: DebugUtilsMessengerCallbackFn,
    p_user_data: ?*anyopaque = null,
};

pub const DebugUtilsMessengerEXT = *opaque {};

/// Vulkan object type enumeration (VK_OBJECT_TYPE_*)
pub const VkObjectType = enum(u32) {
    unknown = 0,
    instance = 1,
    physical_device = 2,
    device = 3,
    queue = 4,
    semaphore = 5,
    command_buffer = 6,
    fence = 7,
    device_memory = 8,
    buffer = 9,
    image = 10,
    event = 11,
    query_pool = 12,
    buffer_view = 13,
    image_view = 14,
    shader_module = 15,
    pipeline_cache = 16,
    pipeline_layout = 17,
    render_pass = 18,
    pipeline = 19,
    descriptor_set_layout = 20,
    sampler = 21,
    descriptor_pool = 22,
    descriptor_set = 23,
    framebuffer = 24,
    command_pool = 25,
    // Extensions
    sampler_ycbcr_conversion = 1000156000,
    descriptor_update_template = 1000085000,
    surface_khr = 1000000000,
    swapchain_khr = 1000001000,
    display_khr = 1000002000,
    display_mode_khr = 1000002001,
    debug_utils_messenger_ext = 1000128000,
    acceleration_structure_khr = 1000150000,
    validation_cache_ext = 1000160000,
    performance_configuration_intel = 1000210000,
    deferred_operation_khr = 1000268000,
    indirect_commands_layout_nv = 1000277000,
    buffer_collection_fuchsia = 1000366000,
    _,
};

/// Debug object name info (VkDebugUtilsObjectNameInfoEXT)
pub const DebugUtilsObjectNameInfoEXT = extern struct {
    s_type: StructureType = .debug_utils_object_name_info_ext,
    p_next: ?*const anyopaque = null,
    object_type: VkObjectType,
    object_handle: u64,
    p_object_name: ?[*:0]const u8,
};

/// Debug label for command buffers (VkDebugUtilsLabelEXT)
pub const DebugUtilsLabelEXT = extern struct {
    s_type: StructureType = .debug_utils_label_ext,
    p_next: ?*const anyopaque = null,
    p_label_name: ?[*:0]const u8,
    color: [4]f32 = .{ 1.0, 1.0, 1.0, 1.0 },
};

/// Object types for debug naming
pub const ObjectType = enum(i32) {
    unknown = 0,
    instance = 1,
    physical_device = 2,
    device = 3,
    queue = 4,
    semaphore = 5,
    command_buffer = 6,
    fence = 7,
    device_memory = 8,
    buffer = 9,
    image = 10,
    event = 11,
    query_pool = 12,
    buffer_view = 13,
    image_view = 14,
    shader_module = 15,
    pipeline_cache = 16,
    pipeline_layout = 17,
    render_pass = 18,
    pipeline = 19,
    descriptor_set_layout = 20,
    sampler = 21,
    descriptor_pool = 22,
    descriptor_set = 23,
    framebuffer = 24,
    command_pool = 25,
    descriptor_update_template = 1000085000,
    _,
};

/// Debug utils object tag info
pub const DebugUtilsObjectTagInfoEXT = extern struct {
    s_type: StructureType = .debug_utils_object_tag_info_ext,
    p_next: ?*const anyopaque = null,
    object_type: ObjectType,
    object_handle: u64,
    tag_name: u64,
    tag_size: usize,
    p_tag: ?*const anyopaque,
};

// =============================================================================
// Subgroup Operations (Vulkan 1.1+)
// =============================================================================

/// Subgroup feature flags
pub const SubgroupFeatureFlags = packed struct(u32) {
    basic: bool = false,
    vote: bool = false,
    arithmetic: bool = false,
    ballot: bool = false,
    shuffle: bool = false,
    shuffle_relative: bool = false,
    clustered: bool = false,
    quad: bool = false,
    partitioned_nv: bool = false,
    _padding: u23 = 0,
};

/// Physical device subgroup properties
pub const PhysicalDeviceSubgroupProperties = extern struct {
    s_type: StructureType = .physical_device_subgroup_properties,
    p_next: ?*anyopaque = null,
    subgroup_size: u32 = 0,
    supported_stages: ShaderStageFlags = .{},
    supported_operations: SubgroupFeatureFlags = .{},
    quad_operations_in_all_stages: u32 = 0,
};

// =============================================================================
// Descriptor Indexing (VK_EXT_descriptor_indexing / Vulkan 1.2)
// =============================================================================

/// Descriptor binding flags
pub const DescriptorBindingFlags = packed struct(u32) {
    update_after_bind: bool = false,
    update_unused_while_pending: bool = false,
    partially_bound: bool = false,
    variable_descriptor_count: bool = false,
    _padding: u28 = 0,
};

/// Descriptor set layout binding flags create info
pub const DescriptorSetLayoutBindingFlagsCreateInfo = extern struct {
    s_type: StructureType = .descriptor_set_layout_binding_flags_create_info,
    p_next: ?*const anyopaque = null,
    binding_count: u32 = 0,
    p_binding_flags: ?[*]const DescriptorBindingFlags = null,
};

/// Physical device descriptor indexing features
pub const PhysicalDeviceDescriptorIndexingFeatures = extern struct {
    s_type: StructureType = .physical_device_descriptor_indexing_features,
    p_next: ?*anyopaque = null,
    shader_input_attachment_array_dynamic_indexing: u32 = 0,
    shader_uniform_texel_buffer_array_dynamic_indexing: u32 = 0,
    shader_storage_texel_buffer_array_dynamic_indexing: u32 = 0,
    shader_uniform_buffer_array_non_uniform_indexing: u32 = 0,
    shader_sampled_image_array_non_uniform_indexing: u32 = 0,
    shader_storage_buffer_array_non_uniform_indexing: u32 = 0,
    shader_storage_image_array_non_uniform_indexing: u32 = 0,
    shader_input_attachment_array_non_uniform_indexing: u32 = 0,
    shader_uniform_texel_buffer_array_non_uniform_indexing: u32 = 0,
    shader_storage_texel_buffer_array_non_uniform_indexing: u32 = 0,
    descriptor_binding_uniform_buffer_update_after_bind: u32 = 0,
    descriptor_binding_sampled_image_update_after_bind: u32 = 0,
    descriptor_binding_storage_image_update_after_bind: u32 = 0,
    descriptor_binding_storage_buffer_update_after_bind: u32 = 0,
    descriptor_binding_uniform_texel_buffer_update_after_bind: u32 = 0,
    descriptor_binding_storage_texel_buffer_update_after_bind: u32 = 0,
    descriptor_binding_update_after_bind_pool: u32 = 0,
    descriptor_binding_update_unused_while_pending: u32 = 0,
    descriptor_binding_partially_bound: u32 = 0,
    descriptor_binding_variable_descriptor_count: u32 = 0,
    runtime_descriptor_array: u32 = 0,
};

/// Physical device descriptor indexing properties
pub const PhysicalDeviceDescriptorIndexingProperties = extern struct {
    s_type: StructureType = .physical_device_descriptor_indexing_properties,
    p_next: ?*anyopaque = null,
    max_update_after_bind_descriptors_in_all_pools: u32 = 0,
    shader_uniform_buffer_array_non_uniform_indexing_native: u32 = 0,
    shader_sampled_image_array_non_uniform_indexing_native: u32 = 0,
    shader_storage_buffer_array_non_uniform_indexing_native: u32 = 0,
    shader_storage_image_array_non_uniform_indexing_native: u32 = 0,
    shader_input_attachment_array_non_uniform_indexing_native: u32 = 0,
    robust_buffer_access_update_after_bind: u32 = 0,
    quad_divergent_implicit_lod: u32 = 0,
    max_per_stage_descriptor_update_after_bind_samplers: u32 = 0,
    max_per_stage_descriptor_update_after_bind_uniform_buffers: u32 = 0,
    max_per_stage_descriptor_update_after_bind_storage_buffers: u32 = 0,
    max_per_stage_descriptor_update_after_bind_sampled_images: u32 = 0,
    max_per_stage_descriptor_update_after_bind_storage_images: u32 = 0,
    max_per_stage_descriptor_update_after_bind_input_attachments: u32 = 0,
    max_per_stage_update_after_bind_resources: u32 = 0,
    max_descriptor_set_update_after_bind_samplers: u32 = 0,
    max_descriptor_set_update_after_bind_uniform_buffers: u32 = 0,
    max_descriptor_set_update_after_bind_uniform_buffers_dynamic: u32 = 0,
    max_descriptor_set_update_after_bind_storage_buffers: u32 = 0,
    max_descriptor_set_update_after_bind_storage_buffers_dynamic: u32 = 0,
    max_descriptor_set_update_after_bind_sampled_images: u32 = 0,
    max_descriptor_set_update_after_bind_storage_images: u32 = 0,
    max_descriptor_set_update_after_bind_input_attachments: u32 = 0,
};

/// Variable descriptor count allocate info
pub const DescriptorSetVariableDescriptorCountAllocateInfo = extern struct {
    s_type: StructureType = .descriptor_set_variable_descriptor_count_allocate_info,
    p_next: ?*const anyopaque = null,
    descriptor_set_count: u32 = 0,
    p_descriptor_counts: ?[*]const u32 = null,
};

// =============================================================================
// Ray Tracing (VK_KHR_ray_tracing_pipeline)
// =============================================================================

/// Ray tracing shader group type
pub const RayTracingShaderGroupTypeKHR = enum(u32) {
    general = 0,
    triangles_hit_group = 1,
    procedural_hit_group = 2,
    _,
};

/// Acceleration structure type
pub const AccelerationStructureTypeKHR = enum(u32) {
    top_level = 0,
    bottom_level = 1,
    generic = 2,
    _,
};

/// Acceleration structure build type
pub const AccelerationStructureBuildTypeKHR = enum(u32) {
    host = 0,
    device = 1,
    host_or_device = 2,
    _,
};

/// Geometry type
pub const GeometryTypeKHR = enum(u32) {
    triangles = 0,
    aabbs = 1,
    instances = 2,
    _,
};

/// Geometry flags
pub const GeometryFlagsKHR = packed struct(u32) {
    is_opaque: bool = false,
    no_duplicate_any_hit_invocation: bool = false,
    _padding: u30 = 0,
};

/// Build acceleration structure flags
pub const BuildAccelerationStructureFlagsKHR = packed struct(u32) {
    allow_update: bool = false,
    allow_compaction: bool = false,
    prefer_fast_trace: bool = false,
    prefer_fast_build: bool = false,
    low_memory: bool = false,
    motion_nv: bool = false,
    _padding: u26 = 0,
};

/// Physical device ray tracing pipeline features
pub const PhysicalDeviceRayTracingPipelineFeaturesKHR = extern struct {
    s_type: StructureType = .physical_device_ray_tracing_pipeline_features_khr,
    p_next: ?*anyopaque = null,
    ray_tracing_pipeline: u32 = 0,
    ray_tracing_pipeline_shader_group_handle_capture_replay: u32 = 0,
    ray_tracing_pipeline_shader_group_handle_capture_replay_mixed: u32 = 0,
    ray_tracing_pipeline_trace_rays_indirect: u32 = 0,
    ray_traversal_primitive_culling: u32 = 0,
};

/// Physical device ray tracing pipeline properties
pub const PhysicalDeviceRayTracingPipelinePropertiesKHR = extern struct {
    s_type: StructureType = .physical_device_ray_tracing_pipeline_properties_khr,
    p_next: ?*anyopaque = null,
    shader_group_handle_size: u32 = 0,
    max_ray_recursion_depth: u32 = 0,
    max_shader_group_stride: u32 = 0,
    shader_group_base_alignment: u32 = 0,
    shader_group_handle_capture_replay_size: u32 = 0,
    max_ray_dispatch_invocation_count: u32 = 0,
    shader_group_handle_alignment: u32 = 0,
    max_ray_hit_attribute_size: u32 = 0,
};

/// Acceleration structure KHR handle
pub const AccelerationStructureKHR = *opaque {};

/// Ray tracing shader group create info
pub const RayTracingShaderGroupCreateInfoKHR = extern struct {
    s_type: StructureType = .ray_tracing_shader_group_create_info_khr,
    p_next: ?*const anyopaque = null,
    type: RayTracingShaderGroupTypeKHR = .general,
    general_shader: u32 = 0xFFFFFFFF,
    closest_hit_shader: u32 = 0xFFFFFFFF,
    any_hit_shader: u32 = 0xFFFFFFFF,
    intersection_shader: u32 = 0xFFFFFFFF,
    p_shader_group_capture_replay_handle: ?*const anyopaque = null,
};

/// Strided device address region for shader binding table
pub const StridedDeviceAddressRegionKHR = extern struct {
    device_address: u64 = 0,
    stride: u64 = 0,
    size: u64 = 0,
};

/// Trace rays indirect command
pub const TraceRaysIndirectCommandKHR = extern struct {
    width: u32,
    height: u32,
    depth: u32,
};

// =============================================================================
// Video Queue (VK_KHR_video_queue)
// =============================================================================

/// Video codec operation flags
pub const VideoCodecOperationFlagsKHR = packed struct(u32) {
    none: bool = false,
    encode_h264: bool = false,
    encode_h265: bool = false,
    decode_h264: bool = false,
    decode_h265: bool = false,
    decode_av1: bool = false,
    _padding: u26 = 0,
};

/// Video session KHR handle
pub const VideoSessionKHR = *opaque {};

/// Video session parameters KHR handle
pub const VideoSessionParametersKHR = *opaque {};

/// Queue family video properties
pub const QueueFamilyVideoPropertiesKHR = extern struct {
    s_type: StructureType = .queue_family_video_properties_khr,
    p_next: ?*anyopaque = null,
    video_codec_operations: VideoCodecOperationFlagsKHR = .{},
};

/// Video coding control flags
pub const VideoCodingControlFlagsKHR = packed struct(u32) {
    reset: bool = false,
    encode_rate_control: bool = false,
    encode_rate_control_layer: bool = false,
    _padding: u29 = 0,
};

// =============================================================================
// Cooperative Matrices (VK_KHR_cooperative_matrix)
// =============================================================================

/// Component type for cooperative matrices
pub const ComponentTypeKHR = enum(u32) {
    float16 = 0,
    float32 = 1,
    float64 = 2,
    sint8 = 3,
    sint16 = 4,
    sint32 = 5,
    sint64 = 6,
    uint8 = 7,
    uint16 = 8,
    uint32 = 9,
    uint64 = 10,
    _,
};

/// Scope for cooperative matrices
pub const ScopeKHR = enum(u32) {
    device = 1,
    workgroup = 2,
    subgroup = 3,
    queue_family = 5,
    _,
};

/// Cooperative matrix use
pub const CooperativeMatrixUseKHR = enum(u32) {
    a = 0,
    b = 1,
    accumulator = 2,
    _,
};

/// Physical device cooperative matrix features
pub const PhysicalDeviceCooperativeMatrixFeaturesKHR = extern struct {
    s_type: StructureType = .physical_device_cooperative_matrix_features_khr,
    p_next: ?*anyopaque = null,
    cooperative_matrix: u32 = 0,
    cooperative_matrix_robust_buffer_access: u32 = 0,
};

/// Cooperative matrix properties
pub const CooperativeMatrixPropertiesKHR = extern struct {
    s_type: StructureType = .cooperative_matrix_properties_khr,
    p_next: ?*anyopaque = null,
    m_size: u32 = 0,
    n_size: u32 = 0,
    k_size: u32 = 0,
    a_type: ComponentTypeKHR = .float32,
    b_type: ComponentTypeKHR = .float32,
    c_type: ComponentTypeKHR = .float32,
    result_type: ComponentTypeKHR = .float32,
    saturating_accumulation: u32 = 0,
    scope: ScopeKHR = .subgroup,
};

// =============================================================================
// Memory Budget Extension (VK_EXT_memory_budget)
// =============================================================================

/// Physical device memory properties 2 (Vulkan 1.1+)
pub const PhysicalDeviceMemoryProperties2 = extern struct {
    s_type: StructureType = .physical_device_memory_properties_2,
    p_next: ?*anyopaque = null,
    memory_properties: PhysicalDeviceMemoryProperties = undefined,
};

/// Memory budget properties for each heap (VK_EXT_memory_budget)
pub const PhysicalDeviceMemoryBudgetPropertiesEXT = extern struct {
    s_type: StructureType = .physical_device_memory_budget_properties_ext,
    p_next: ?*anyopaque = null,
    /// Budget available for each heap (in bytes)
    heap_budget: [VK_MAX_MEMORY_HEAPS]u64 = [_]u64{0} ** VK_MAX_MEMORY_HEAPS,
    /// Current usage for each heap (in bytes)
    heap_usage: [VK_MAX_MEMORY_HEAPS]u64 = [_]u64{0} ** VK_MAX_MEMORY_HEAPS,
};

// =============================================================================
// Push Descriptor Extension (VK_KHR_push_descriptor)
// =============================================================================

/// Push descriptor properties (VK_KHR_push_descriptor)
pub const PhysicalDevicePushDescriptorPropertiesKHR = extern struct {
    s_type: StructureType = .physical_device_push_descriptor_properties_khr,
    p_next: ?*anyopaque = null,
    /// Maximum number of push descriptors
    max_push_descriptors: u32 = 0,
};
