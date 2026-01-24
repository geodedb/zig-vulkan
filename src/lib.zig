// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Zig Vulkan Compute Library
//!
//! A production-grade GPU compute library for Zig, providing a clean API for
//! Vulkan compute operations. Designed for ML workloads and GPU-heavy applications.
//!
//! ## Features
//!
//! - **Dynamic Vulkan loader** - works without Vulkan SDK installation
//! - **Graceful degradation** - fails gracefully when Vulkan is unavailable
//! - **Cross-platform** - supports Linux, Windows, and macOS
//! - **Compute-focused** - optimized for ML and scientific computing
//! - **Safe wrappers** - explicit resource management with defer
//! - **Debug validation** - SPIR-V and buffer size validation in debug builds
//! - **Comprehensive errors** - detailed error types with descriptions
//!
//! ## Quick Start
//!
//! ```zig
//! const vulkan = @import("vulkan");
//!
//! pub fn main() !void {
//!     // Check if Vulkan is available
//!     if (!vulkan.isAvailable()) {
//!         return error.VulkanUnavailable;
//!     }
//!
//!     // Create a compute context
//!     var ctx = try vulkan.Context.init(allocator, .{
//!         .enable_validation = @import("builtin").mode == .Debug,
//!         .enable_debug_utils = @import("builtin").mode == .Debug,
//!     });
//!     defer ctx.deinit();
//!
//!     // Create buffers
//!     var input = try vulkan.Buffer.init(&ctx, 1024, .staging);
//!     defer input.deinit(&ctx);
//!
//!     var output = try vulkan.Buffer.init(&ctx, 1024, .storage);
//!     defer output.deinit(&ctx);
//!
//!     // Load shader and create pipeline
//!     var shader = try vulkan.ShaderModule.initFromBytes(&ctx, @embedFile("compute.spv"));
//!     defer shader.deinit();
//!
//!     // ... dispatch compute work
//! }
//! ```
//!
//! ## System Probing
//!
//! Before creating a context, you can probe the system for available devices:
//!
//! ```zig
//! var result = vulkan.probe(allocator);
//! defer result.deinit(allocator);
//!
//! if (result.isAvailable()) {
//!     for (result.devices) |dev| {
//!         std.debug.print("Device: {s}\n", .{dev.getName()});
//!     }
//! }
//! ```
//!
//! ## Buffer Types
//!
//! | Type | Memory | Use Case |
//! |------|--------|----------|
//! | `.storage` | Device-local | GPU compute read/write |
//! | `.uniform` | Host-visible | Constant data |
//! | `.staging` | Host-visible | CPU to GPU uploads |
//! | `.readback` | Host-cached | GPU to CPU downloads |
//!
//! ## Module Organization
//!
//! - `vk` - Low-level Vulkan bindings and types
//! - `compute` - High-level compute abstractions (Context, Buffer, Pipeline)
//! - `util` - Utilities (memory allocation, validation, alignment)
//!
//! ## Callback Ownership Conventions
//!
//! This library uses callbacks in several APIs. The following ownership
//! conventions apply to all callback types:
//!
//! ### Lifetime Requirements
//!
//! All callbacks must remain valid for the duration they are registered:
//!
//! | Callback Type | Lifetime | Ownership |
//! |---------------|----------|-----------|
//! | `ErrorHandler` | Until handler is replaced/removed | Borrowed |
//! | `ResourceRecreateCallback` | Until resource is unregistered | Borrowed |
//! | `ChangeCallback` (hot reload) | Until watcher is stopped | Borrowed |
//! | `MemoryStatusCallback` | Until monitor is stopped | Borrowed |
//! | `EvictionCallback` | Until eviction policy removed | Borrowed |
//! | `TaskCallback` (graph) | Until graph execution completes | Borrowed |
//! | `DebugUtilsMessengerCallbackFn` | Until messenger is destroyed | Borrowed |
//!
//! ### user_data Parameters
//!
//! Many callbacks accept a `user_data: ?*anyopaque` parameter:
//!
//! - The library **does not** take ownership of user_data
//! - The library **does not** free user_data memory
//! - The caller must ensure user_data remains valid while the callback is registered
//! - user_data is passed through unchanged to the callback
//!
//! ### Memory Passed to Callbacks
//!
//! Data passed *to* callbacks follows these rules:
//!
//! - Slices (e.g., `[]const u8`) are **borrowed** and valid only during the callback
//! - Copy the data if you need it after the callback returns
//! - Never store pointers to callback-provided data without copying
//!
//! ### Thread Safety
//!
//! Callbacks may be invoked from different threads:
//!
//! - `DebugUtilsMessengerCallbackFn`: May be called from any Vulkan thread
//! - `EvictionCallback`: Called from the thread performing eviction
//! - `ChangeCallback`: Called from the file watcher thread
//! - Other callbacks: Called from the thread that initiated the operation
//!
//! See individual callback documentation for specific thread-safety requirements.
//!
//! ## Thread Safety Overview
//!
//! This library provides limited thread-safety. Most types are NOT thread-safe
//! by default. Here's a summary of thread-safety guarantees:
//!
//! ### Thread-Safe Operations
//!
//! | Type/Function | Operation | Notes |
//! |---------------|-----------|-------|
//! | `isAvailable()` | Checking Vulkan availability | Read-only |
//! | `probe()` | System probing | Creates temporary resources |
//! | `Context` (read-only) | `deviceInfo()`, `deviceExtensions()` | After init |
//! | `Context` | Queue submission | With Vulkan synchronization |
//!
//! ### Not Thread-Safe
//!
//! | Type | Notes |
//! |------|-------|
//! | `Buffer` | All operations require external sync |
//! | `CommandBuffer` | One command buffer per thread |
//! | `DescriptorPool` | Pool allocation not thread-safe |
//! | `ComputePipeline` | Creation not thread-safe; use is safe |
//!
//! ### Multi-Threading Guidelines
//!
//! 1. **One Context, Many Threads**: Create one Context at startup,
//!    share it across threads for read-only operations.
//!
//! 2. **Command Buffer Per Thread**: Create separate CommandBuffers
//!    for each thread to avoid synchronization overhead.
//!
//! 3. **Vulkan Synchronization**: Use fences and semaphores for
//!    GPU-CPU and GPU-GPU synchronization.
//!
//! 4. **Resource Lifetime**: Ensure resources are not destroyed
//!    while in use by GPU or other threads.
//!
//! 5. **Queue Submission**: Multiple threads can submit to the same
//!    queue if submissions are properly synchronized with semaphores.

const std = @import("std");

// Low-level Vulkan module
pub const vk = @import("vk/mod.zig");

// High-level compute module
pub const compute = @import("compute/mod.zig");

// Utility module
pub const util = @import("util/mod.zig");

// Standalone feature modules
pub const descriptor_template = @import("descriptor/template.zig");
pub const usage_pools = @import("memory/usage_pools.zig");
pub const batch_command = @import("command/batch.zig");
pub const shader_variants = @import("pipeline/variants.zig");

// =============================================================================
// Top-level API (convenience re-exports)
// =============================================================================

/// Check if Vulkan is available on this system
pub const isAvailable = vk.isAvailable;

/// Probe the system for Vulkan availability and device information
pub const probe = compute.Context.probe;

// Main types
pub const Context = compute.Context;
pub const Options = compute.Options;
pub const ProbeResult = compute.ProbeResult;
pub const DeviceInfo = compute.DeviceInfo;

// Debug utilities
pub const DebugStats = compute.context.DebugStats;
pub const DebugSeverity = compute.context.DebugSeverity;
pub const DebugMessageType = compute.context.DebugMessageType;

pub const Buffer = compute.Buffer;
pub const BufferUsage = compute.BufferUsage;
pub const Transfer = compute.Transfer;

pub const ShaderModule = compute.ShaderModule;
pub const DescriptorSetLayout = compute.DescriptorSetLayout;
pub const DescriptorPool = compute.DescriptorPool;
pub const DescriptorBinding = compute.DescriptorBinding;
pub const ComputePipeline = compute.ComputePipeline;
pub const PushConstantRange = compute.PushConstantRange;
pub const SpecializationEntry = compute.SpecializationEntry;
pub const SpecializationConfig = compute.SpecializationConfig;
pub const PipelineCache = compute.PipelineCache;
pub const PersistentPipelineCache = compute.PersistentPipelineCache;
pub const PersistentCacheConfig = compute.PersistentCacheConfig;
pub const PipelineCacheStats = compute.PipelineCacheStats;
pub const updateDescriptorSet = compute.updateDescriptorSet;

pub const CommandBuffer = compute.CommandBuffer;
pub const Fence = compute.Fence;
pub const Semaphore = compute.Semaphore;
pub const Submission = compute.Submission;
pub const TimestampQuery = compute.TimestampQuery;
pub const PipelineStatisticsQuery = compute.PipelineStatisticsQuery;
pub const PipelineStatisticsFlags = compute.PipelineStatisticsFlags;
pub const PipelineStatistics = compute.PipelineStatistics;

// Compute dispatch utilities
pub const ComputeDispatcher = compute.ComputeDispatcher;
pub const WorkgroupSize = compute.WorkgroupSize;
pub const DispatchDimensions = compute.DispatchDimensions;
pub const DispatchParams = compute.DispatchParams;
pub const DispatchBatch = compute.DispatchBatch;

// Multi-queue support
pub const QueueManager = compute.QueueManager;
pub const QueueFamilyInfo = compute.QueueFamilyInfo;
pub const QueueType = compute.QueueType;
pub const QueueCapabilities = compute.QueueCapabilities;
pub const TimelineSemaphore = compute.TimelineSemaphore;
pub const AsyncTransfer = compute.AsyncTransfer;
pub const QueueSync = compute.QueueSync;
pub const CrossQueueCoordinator = compute.CrossQueueCoordinator;
pub const BufferOwnershipTransfer = compute.BufferOwnershipTransfer;
pub const ImageOwnershipTransfer = compute.ImageOwnershipTransfer;

// Multi-device support
pub const MultiDeviceContext = compute.MultiDeviceContext;
pub const PhysicalDeviceHandle = compute.PhysicalDeviceHandle;
pub const DeviceSelector = compute.DeviceSelector;
pub const DeviceCapabilities = compute.DeviceCapabilities;
pub const DevicePool = compute.DevicePool;

// Extensions support (subgroup, descriptor indexing, ray tracing, video, cooperative matrices)
pub const extensions = compute.extensions;
pub const SubgroupCapabilities = compute.SubgroupCapabilities;
pub const SubgroupOperation = compute.SubgroupOperation;
pub const DescriptorIndexingCapabilities = compute.DescriptorIndexingCapabilities;
pub const RayTracingCapabilities = compute.RayTracingCapabilities;
pub const VideoCapabilities = compute.VideoCapabilities;
pub const CooperativeMatrixCapabilities = compute.CooperativeMatrixCapabilities;
pub const ExtensionCapabilities = compute.ExtensionCapabilities;
pub const querySubgroupProperties = compute.querySubgroupProperties;
pub const queryDescriptorIndexingFeatures = compute.queryDescriptorIndexingFeatures;
pub const queryRayTracingCapabilities = compute.queryRayTracingCapabilities;
pub const queryVideoCapabilities = compute.queryVideoCapabilities;
pub const queryCooperativeMatrixCapabilities = compute.queryCooperativeMatrixCapabilities;
pub const queryAllCapabilities = compute.queryAllCapabilities;

// Extended extensions (sparse, mesh, debug, memory budget, push descriptors)
pub const SparseResourceCapabilities = compute.SparseResourceCapabilities;
pub const MeshShaderCapabilities = compute.MeshShaderCapabilities;
pub const DebugPrintfCapabilities = compute.DebugPrintfCapabilities;
pub const MemoryHeapBudget = compute.MemoryHeapBudget;
pub const MemoryBudgetInfo = compute.MemoryBudgetInfo;
pub const MemoryBudgetMonitor = compute.MemoryBudgetMonitor;
pub const MemoryStatusCallback = compute.MemoryStatusCallback;
pub const MemoryPressureHandler = compute.MemoryPressureHandler;
pub const MemoryPressureConfig = compute.MemoryPressureConfig;
pub const EvictionCallback = compute.EvictionCallback;
pub const EvictionResult = compute.EvictionResult;
pub const PressureStats = compute.PressureStats;
pub const PushDescriptorCapabilities = compute.PushDescriptorCapabilities;
pub const ExtendedCapabilities = compute.ExtendedCapabilities;
pub const querySparseResourceCapabilities = compute.querySparseResourceCapabilities;
pub const queryMeshShaderCapabilities = compute.queryMeshShaderCapabilities;
pub const queryDebugPrintfCapabilities = compute.queryDebugPrintfCapabilities;
pub const queryMemoryBudget = compute.queryMemoryBudget;
pub const queryPushDescriptorCapabilities = compute.queryPushDescriptorCapabilities;
pub const queryExtendedCapabilities = compute.queryExtendedCapabilities;

// Error types
pub const Error = vk.Error;
pub const ErrorContext = vk.ErrorContext;
pub const ErrorTrace = vk.ErrorTrace;
pub const ErrorLog = vk.ErrorLog;
pub const SourceLocation = vk.SourceLocation;
pub const Diagnostics = vk.Diagnostics;
pub const describeError = vk.describe;
pub const logError = vk.logError;
pub const logVkError = vk.logVkError;

// Memory utilities
pub const MemoryUsage = util.MemoryUsage;
pub const Alignment = util.Alignment;

// Pooled memory allocator (efficient suballocation)
pub const PooledAllocator = util.PooledAllocator;
pub const PoolConfig = util.PoolConfig;
pub const Suballocation = util.Suballocation;
pub const AllocatorStats = util.AllocatorStats;
pub const PooledBuffer = compute.PooledBuffer;
pub const PooledTransfer = compute.PooledTransfer;

// Validation utilities
pub const validation = util.validation;
pub const SpirvValidationResult = util.SpirvValidationResult;
pub const SpirvInfo = util.SpirvInfo;
pub const validateSpirv = util.validateSpirv;
pub const validateSpirvDetailed = util.validateSpirvDetailed;

// GLSL compilation
pub const glsl = util.glsl;

// Error recovery
pub const recovery = util.recovery;
pub const DeviceLostHandler = util.DeviceLostHandler;
pub const DeviceLostConfig = util.DeviceLostConfig;
pub const DeviceLostStats = util.DeviceLostStats;
pub const RecreationError = util.RecreationError;
pub const ErrorSeverity = util.ErrorSeverity;
pub const ErrorCollector = util.ErrorCollector;
pub const withDeviceRecovery = util.withDeviceRecovery;
pub const DeviceLostChecker = util.DeviceLostChecker;
pub const GlobalDeviceLostHandler = util.GlobalDeviceLostHandler;
pub const withDeviceRecoverySimple = util.withDeviceRecoverySimple;

// Shader hot-reload (development workflow)
pub const hot_reload = util.hot_reload;
pub const ShaderWatcher = util.ShaderWatcher;
pub const HotReloadShader = util.HotReloadShader;
pub const ShaderFileInfo = util.ShaderFileInfo;
pub const WatchOptions = util.WatchOptions;

// Vulkan types (for advanced usage)
pub const DescriptorType = vk.DescriptorType;
pub const PipelineStageFlags = vk.PipelineStageFlags;
pub const AccessFlags = vk.AccessFlags;
pub const DispatchIndirectCommand = vk.DispatchIndirectCommand;

// Extended descriptor update functions
pub const updateDescriptorSetEx = compute.updateDescriptorSetEx;
pub const updateDescriptorSetAdvanced = compute.updateDescriptorSetAdvanced;
pub const DescriptorBufferBinding = compute.DescriptorBufferBinding;
pub const MAX_DESCRIPTOR_BINDINGS = compute.MAX_DESCRIPTOR_BINDINGS;

// Compute profiling
pub const ComputeProfiler = compute.ComputeProfiler;
pub const ProfileResult = compute.ProfileResult;
pub const ProfileStats = compute.ProfileStats;
pub const ProfileScope = compute.ProfileScope;

// Async compute coordination
pub const AsyncCoordinator = compute.AsyncCoordinator;
pub const ResourceState = compute.ResourceState;
pub const FrameSync = compute.FrameSync;
pub const StreamBuffer = compute.StreamBuffer;
pub const SyncManager = compute.SyncManager;
pub const SyncPoint = compute.SyncPoint;
pub const SyncEvent = compute.SyncEvent;
pub const GpuEvent = compute.GpuEvent;

// Compute graph (DAG-based scheduling)
pub const ComputeGraph = compute.ComputeGraph;
pub const TaskNode = compute.TaskNode;
pub const GraphResourceHandle = compute.ResourceHandle;
pub const GraphResourceUsage = compute.ResourceUsage;
pub const GraphAccessMode = compute.AccessMode;
pub const GraphExecutionResult = compute.ExecutionResult;
pub const GraphSubmitOptions = compute.SubmitOptions;

// Shader reflection
pub const ShaderReflection = util.ShaderReflection;
pub const ReflectedBinding = util.ReflectedBinding;
pub const PushConstantInfo = util.PushConstantInfo;

// Resource aliasing
pub const AliasingPool = util.AliasingPool;
pub const ResourceLifetime = util.ResourceLifetime;
pub const AliasedResource = util.AliasedResource;

// Debug labels (RenderDoc/NSight integration)
pub const DebugUtils = util.DebugUtils;
pub const DebugCategory = util.DebugCategory;
pub const DebugColors = util.DebugColors;
pub const ScopedLabel = util.ScopedLabel;
pub const LabeledRecorder = util.LabeledRecorder;

// Descriptor set templates
pub const UpdateTemplate = descriptor_template.UpdateTemplate;
pub const TemplateBuilder = descriptor_template.TemplateBuilder;
pub const TemplateCache = descriptor_template.TemplateCache;
pub const UpdateDataBuilder = descriptor_template.UpdateDataBuilder;
pub const PushDescriptorUpdater = descriptor_template.PushDescriptorUpdater;

// Memory pools by usage pattern
pub const RingBufferPool = usage_pools.RingBufferPool;
pub const BlockPool = usage_pools.BlockPool;
pub const LinearPool = usage_pools.LinearPool;
pub const PoolManager = usage_pools.PoolManager;
pub const UsagePattern = usage_pools.UsagePattern;

// Batch command recording
pub const BatchRecorder = batch_command.BatchRecorder;
pub const ReusableCommandBuffer = batch_command.ReusableCommandBuffer;
pub const ResourceAccess = batch_command.ResourceAccess;
pub const DispatchConfig = batch_command.DispatchConfig;

// Shader variants (specialization constant caching)
pub const ShaderVariantManager = shader_variants.ShaderVariantManager;
pub const VariantBuilder = shader_variants.VariantBuilder;
pub const VariantSystem = shader_variants.VariantSystem;
pub const SpecConstantDef = shader_variants.SpecConstantDef;
pub const SpecValue = shader_variants.SpecValue;
pub const SpecType = shader_variants.SpecType;

// =============================================================================
// Testing Utilities
// =============================================================================

/// Testing utilities for unit testing without GPU hardware
pub const testing = @import("testing/mod.zig");
pub const MockContext = testing.MockContext;
pub const MockDeviceProperties = testing.MockDeviceProperties;
pub const MockOperation = testing.MockOperation;

// =============================================================================
// Version Information
// =============================================================================

pub const version = std.SemanticVersion{
    .major = 0,
    .minor = 6,
    .patch = 0,
};

pub const version_string = "0.6.0";

// =============================================================================
// Tests
// =============================================================================

test {
    // Run all submodule tests
    std.testing.refAllDecls(@This());
}

test "library version is valid" {
    try std.testing.expect(version.major == 0);
    try std.testing.expect(version.minor == 6);
    try std.testing.expect(version.patch == 0);
}

test "isAvailable does not panic" {
    // This should never panic, regardless of Vulkan availability
    _ = isAvailable();
}
