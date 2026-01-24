// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Vulkan Compute Module
//!
//! High-level abstractions for GPU compute operations.

pub const context = @import("context.zig");
pub const buffer = @import("buffer.zig");
pub const pipeline = @import("pipeline.zig");
pub const command = @import("command.zig");
pub const queue = @import("queue.zig");
pub const multi_device = @import("multi_device.zig");
pub const extensions = @import("extensions.zig");
pub const dispatch = @import("dispatch.zig");
pub const profiling = @import("profiling.zig");
pub const async_compute = @import("async_compute.zig");
pub const graph = @import("graph.zig");

// Re-export main types
pub const Context = context.Context;
pub const Options = context.Options;
pub const ProbeResult = context.ProbeResult;
pub const DeviceInfo = context.DeviceInfo;
pub const DebugStats = context.DebugStats;
pub const DebugSeverity = context.DebugSeverity;
pub const DebugMessageType = context.DebugMessageType;

pub const Buffer = buffer.Buffer;
pub const BufferUsage = buffer.BufferUsage;
pub const Transfer = buffer.Transfer;
pub const PooledBuffer = buffer.PooledBuffer;
pub const PooledTransfer = buffer.PooledTransfer;

pub const ShaderModule = pipeline.ShaderModule;
pub const DescriptorSetLayout = pipeline.DescriptorSetLayout;
pub const DescriptorPool = pipeline.DescriptorPool;
pub const DescriptorBinding = pipeline.DescriptorBinding;
pub const ComputePipeline = pipeline.ComputePipeline;
pub const PushConstantRange = pipeline.PushConstantRange;
pub const SpecializationEntry = pipeline.SpecializationEntry;
pub const SpecializationConfig = pipeline.SpecializationConfig;
pub const PipelineCache = pipeline.PipelineCache;
pub const PersistentPipelineCache = pipeline.PersistentPipelineCache;
pub const PersistentCacheConfig = pipeline.PersistentPipelineCache.PersistentCacheConfig;
pub const PipelineCacheStats = pipeline.PersistentPipelineCache.CacheStats;
pub const updateDescriptorSet = pipeline.updateDescriptorSet;
pub const updateDescriptorSetEx = pipeline.updateDescriptorSetEx;
pub const updateDescriptorSetAdvanced = pipeline.updateDescriptorSetAdvanced;
pub const DescriptorBufferBinding = pipeline.DescriptorBufferBinding;
pub const MAX_DESCRIPTOR_BINDINGS = pipeline.MAX_DESCRIPTOR_BINDINGS;
pub const validateSpirvDetailed = pipeline.validateSpirvDetailed;
pub const HotReloadablePipeline = pipeline.HotReloadablePipeline;
pub const HotReloadConfig = pipeline.HotReloadConfig;

// Compute dispatch helpers
pub const ComputeDispatcher = dispatch.ComputeDispatcher;
pub const WorkgroupSize = dispatch.WorkgroupSize;
pub const DispatchDimensions = dispatch.DispatchDimensions;
pub const DispatchParams = dispatch.DispatchParams;
pub const DispatchBatch = dispatch.DispatchBatch;

pub const CommandBuffer = command.CommandBuffer;
pub const Fence = command.Fence;
pub const Semaphore = command.Semaphore;
pub const Submission = command.Submission;
pub const TimestampQuery = command.TimestampQuery;
pub const PipelineStatisticsQuery = command.PipelineStatisticsQuery;
pub const PipelineStatisticsFlags = command.PipelineStatisticsFlags;
pub const PipelineStatistics = command.PipelineStatistics;

// Queue management (multi-queue support)
pub const QueueManager = queue.QueueManager;
pub const QueueFamilyInfo = queue.QueueFamilyInfo;
pub const QueueType = queue.QueueType;
pub const QueueCapabilities = queue.QueueCapabilities;
pub const TimelineSemaphore = queue.TimelineSemaphore;
pub const AsyncTransfer = queue.AsyncTransfer;
pub const QueueSync = queue.QueueSync;
pub const CrossQueueCoordinator = queue.CrossQueueCoordinator;
pub const BufferOwnershipTransfer = queue.BufferOwnershipTransfer;
pub const ImageOwnershipTransfer = queue.ImageOwnershipTransfer;

// Multi-device support
pub const MultiDeviceContext = multi_device.MultiDeviceContext;
pub const PhysicalDeviceHandle = multi_device.PhysicalDeviceHandle;
pub const DeviceSelector = multi_device.DeviceSelector;
pub const DeviceCapabilities = multi_device.DeviceCapabilities;
pub const DevicePool = multi_device.DevicePool;

// Extensions support
pub const SubgroupCapabilities = extensions.SubgroupCapabilities;
pub const SubgroupOperation = extensions.SubgroupOperation;
pub const DescriptorIndexingCapabilities = extensions.DescriptorIndexingCapabilities;
pub const RayTracingCapabilities = extensions.RayTracingCapabilities;
pub const VideoCapabilities = extensions.VideoCapabilities;
pub const CooperativeMatrixCapabilities = extensions.CooperativeMatrixCapabilities;
pub const ExtensionCapabilities = extensions.ExtensionCapabilities;
pub const querySubgroupProperties = extensions.querySubgroupProperties;
pub const queryDescriptorIndexingFeatures = extensions.queryDescriptorIndexingFeatures;
pub const queryRayTracingCapabilities = extensions.queryRayTracingCapabilities;
pub const queryVideoCapabilities = extensions.queryVideoCapabilities;
pub const queryCooperativeMatrixCapabilities = extensions.queryCooperativeMatrixCapabilities;
pub const queryAllCapabilities = extensions.queryAllCapabilities;

// Extended extension support (sparse, mesh, debug, memory budget, push descriptors)
pub const SparseResourceCapabilities = extensions.SparseResourceCapabilities;
pub const MeshShaderCapabilities = extensions.MeshShaderCapabilities;
pub const DebugPrintfCapabilities = extensions.DebugPrintfCapabilities;
pub const MemoryHeapBudget = extensions.MemoryHeapBudget;
pub const MemoryBudgetInfo = extensions.MemoryBudgetInfo;
pub const MemoryBudgetMonitor = extensions.MemoryBudgetMonitor;
pub const MemoryStatusCallback = extensions.MemoryStatusCallback;
pub const MemoryPressureHandler = extensions.MemoryPressureHandler;
pub const MemoryPressureConfig = extensions.MemoryPressureConfig;
pub const EvictionCallback = extensions.EvictionCallback;
pub const EvictionResult = extensions.EvictionResult;
pub const EvictableResource = extensions.EvictableResource;
pub const PressureStats = extensions.MemoryPressureHandler.PressureStats;
pub const PushDescriptorCapabilities = extensions.PushDescriptorCapabilities;
pub const ExtendedCapabilities = extensions.ExtendedCapabilities;
pub const querySparseResourceCapabilities = extensions.querySparseResourceCapabilities;
pub const queryMeshShaderCapabilities = extensions.queryMeshShaderCapabilities;
pub const queryDebugPrintfCapabilities = extensions.queryDebugPrintfCapabilities;
pub const queryMemoryBudget = extensions.queryMemoryBudget;
pub const queryPushDescriptorCapabilities = extensions.queryPushDescriptorCapabilities;
pub const queryExtendedCapabilities = extensions.queryExtendedCapabilities;

// Profiling support
pub const ComputeProfiler = profiling.ComputeProfiler;
pub const ProfileResult = profiling.ProfileResult;
pub const ProfileStats = profiling.ProfileStats;
pub const ProfileScope = profiling.ProfileScope;

// Async compute coordination
pub const AsyncCoordinator = async_compute.AsyncCoordinator;
pub const ResourceState = async_compute.ResourceState;
pub const FrameSync = async_compute.FrameSync;
pub const StreamBuffer = async_compute.StreamBuffer;
pub const OperationType = async_compute.OperationType;
pub const SyncManager = async_compute.SyncManager;
pub const SyncPoint = async_compute.SyncPoint;
pub const SyncEvent = async_compute.SyncEvent;
pub const GpuEvent = async_compute.GpuEvent;
pub const GpuEventBatch = async_compute.GpuEventBatch;

// Compute graph
pub const ComputeGraph = graph.ComputeGraph;
pub const TaskNode = graph.TaskNode;
pub const ResourceHandle = graph.ResourceHandle;
pub const ResourceUsage = graph.ResourceUsage;
pub const AccessMode = graph.AccessMode;
pub const BarrierInfo = graph.BarrierInfo;
pub const TaskQueueHint = graph.TaskQueueHint;

// GPU profiling
pub const GpuProfiler = graph.GpuProfiler;
pub const ProfileResults = graph.ProfileResults;
pub const TaskProfile = graph.TaskProfile;
pub const ExecutionResult = graph.ComputeGraph.ExecutionResult;
pub const SubmitOptions = graph.ComputeGraph.SubmitOptions;

test {
    @import("std").testing.refAllDecls(@This());
}
