// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Utility Modules
//!
//! Helper utilities for memory allocation, validation, and other common operations.
//!
//! ## Modules
//!
//! - `allocator`: Simple GPU memory allocation (one block per allocation)
//! - `pooled_allocator`: Efficient pooled GPU memory allocation with suballocation
//! - `validation`: Runtime and debug-time validation utilities
//! - `glsl`: GLSL to SPIR-V compilation wrapper
//! - `hot_reload`: Shader hot-reload for development workflows

pub const allocator = @import("allocator.zig");
pub const pooled_allocator = @import("pooled_allocator.zig");
pub const validation = @import("validation.zig");
pub const glsl = @import("glsl.zig");
pub const recovery = @import("error_recovery.zig");
pub const hot_reload = @import("hot_reload.zig");
pub const reflection = @import("reflection.zig");
pub const resource_aliasing = @import("resource_aliasing.zig");
pub const debug_labels = @import("debug_labels.zig");

// Re-export simple allocator types (backward compatibility)
pub const SimpleAllocator = allocator.SimpleAllocator;
pub const MemoryUsage = allocator.MemoryUsage;
pub const MemoryBlock = allocator.MemoryBlock;
pub const AllocationStrategy = allocator.AllocationStrategy;
pub const Alignment = allocator.Alignment;

// Re-export pooled allocator types
pub const PooledAllocator = pooled_allocator.PooledAllocator;
pub const PoolConfig = pooled_allocator.PoolConfig;
pub const Suballocation = pooled_allocator.Suballocation;
pub const AllocatorStats = pooled_allocator.AllocatorStats;

// Re-export validation types
pub const SpirvValidationResult = validation.SpirvValidationResult;
pub const SpirvInfo = validation.SpirvInfo;
pub const validateSpirv = validation.validateSpirv;
pub const validateSpirvDetailed = validation.validateSpirvDetailed;
pub const validateBufferSize = validation.validateBufferSize;
pub const BufferType = validation.BufferType;

// Re-export hot-reload types
pub const ShaderWatcher = hot_reload.ShaderWatcher;
pub const HotReloadShader = hot_reload.HotReloadShader;
pub const ShaderFileInfo = hot_reload.ShaderFileInfo;
pub const WatchOptions = hot_reload.WatchOptions;
pub const WatchEntry = hot_reload.WatchEntry;

// Re-export reflection types
pub const ShaderReflection = reflection.ShaderReflection;
pub const ReflectedBinding = reflection.DescriptorBinding;
pub const PushConstantInfo = reflection.PushConstantInfo;
pub const EntryPoint = reflection.EntryPoint;
pub const ReflectedWorkgroupSize = reflection.WorkgroupSize;

// Re-export resource aliasing types
pub const AliasingPool = resource_aliasing.AliasingPool;
pub const ResourceLifetime = resource_aliasing.Lifetime;
pub const AliasedResource = resource_aliasing.AliasedResource;
pub const AliasingStats = resource_aliasing.AliasingStats;

// Re-export debug labels types
pub const DebugUtils = debug_labels.DebugUtils;
pub const DebugCategory = debug_labels.Category;
pub const DebugColors = debug_labels.Colors;
pub const ScopedLabel = debug_labels.ScopedLabel;
pub const LabeledRecorder = debug_labels.LabeledRecorder;

// Re-export error recovery types
pub const DeviceLostHandler = recovery.DeviceLostHandler;
pub const DeviceLostConfig = recovery.DeviceLostConfig;
pub const DeviceLostStats = recovery.DeviceLostStats;
pub const RecreationError = recovery.RecreationError;
pub const RegisteredResource = recovery.RegisteredResource;
pub const ResourceRecreateCallback = recovery.ResourceRecreateCallback;
pub const ErrorSeverity = recovery.ErrorSeverity;
pub const ErrorContext = recovery.ErrorContext;
pub const ErrorCollector = recovery.ErrorCollector;
pub const RetryConfig = recovery.RetryConfig;
pub const RecoveryStrategy = recovery.RecoveryStrategy;
pub const getErrorSeverity = recovery.getErrorSeverity;
pub const isRecoverable = recovery.isRecoverable;
pub const isTransient = recovery.isTransient;
pub const withDeviceRecovery = recovery.withDeviceRecovery;
pub const DeviceLostChecker = recovery.DeviceLostChecker;
pub const GlobalDeviceLostHandler = recovery.GlobalDeviceLostHandler;
pub const withDeviceRecoverySimple = recovery.withDeviceRecoverySimple;

// Re-export shader cache types
pub const ShaderCache = glsl.ShaderCache;
pub const ShaderCacheConfig = glsl.ShaderCacheConfig;
pub const ShaderCacheStats = glsl.ShaderCache.CacheStats;
pub const compileWithCache = glsl.compileWithCache;

test {
    @import("std").testing.refAllDecls(@This());
}
