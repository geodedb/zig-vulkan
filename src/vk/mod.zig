// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Low-level Vulkan Bindings
//!
//! This module provides direct Vulkan bindings with dynamic loading support.

pub const types = @import("types.zig");
pub const loader = @import("loader.zig");
pub const err = @import("error.zig");

// Re-export commonly used types
pub const Instance = types.Instance;
pub const PhysicalDevice = types.PhysicalDevice;
pub const Device = types.Device;
pub const Queue = types.Queue;
pub const CommandPool = types.CommandPool;
pub const CommandBuffer = types.CommandBuffer;
pub const Buffer = types.Buffer;
pub const DeviceMemory = types.DeviceMemory;
pub const ShaderModule = types.ShaderModule;
pub const DescriptorSetLayout = types.DescriptorSetLayout;
pub const DescriptorPool = types.DescriptorPool;
pub const DescriptorSet = types.DescriptorSet;
pub const PipelineLayout = types.PipelineLayout;
pub const Pipeline = types.Pipeline;
pub const Fence = types.Fence;
pub const Semaphore = types.Semaphore;
pub const QueryPool = types.QueryPool;

pub const Result = types.Result;
pub const PhysicalDeviceType = types.PhysicalDeviceType;
pub const DescriptorType = types.DescriptorType;
pub const BufferUsageFlags = types.BufferUsageFlags;
pub const MemoryPropertyFlags = types.MemoryPropertyFlags;
pub const PipelineStageFlags = types.PipelineStageFlags;
pub const AccessFlags = types.AccessFlags;
pub const ShaderStageFlags = types.ShaderStageFlags;
pub const DispatchIndirectCommand = types.DispatchIndirectCommand;
pub const SpecializationMapEntry = types.SpecializationMapEntry;
pub const SpecializationInfo = types.SpecializationInfo;

pub const BaseLoader = loader.BaseLoader;
pub const GlobalDispatch = loader.GlobalDispatch;
pub const InstanceDispatch = loader.InstanceDispatch;
pub const DeviceDispatch = loader.DeviceDispatch;
pub const LoadError = loader.LoadError;
pub const VulkanError = loader.VulkanError;

pub const Error = err.Error;
pub const fromVkResult = err.fromVkResult;
pub const check = err.check;
pub const describe = err.describe;
pub const isRecoverable = err.isRecoverable;
pub const ErrorContext = err.ErrorContext;
pub const ErrorTrace = err.ErrorTrace;
pub const ErrorLog = err.ErrorLog;
pub const SourceLocation = err.SourceLocation;
pub const Diagnostics = err.Diagnostics;
pub const logError = err.logError;
pub const logVkError = err.logVkError;

// Version helpers
pub const API_VERSION_1_0 = types.API_VERSION_1_0;
pub const API_VERSION_1_1 = types.API_VERSION_1_1;
pub const API_VERSION_1_2 = types.API_VERSION_1_2;
pub const API_VERSION_1_3 = types.API_VERSION_1_3;
pub const makeApiVersion = types.makeApiVersion;
pub const apiVersionMajor = types.apiVersionMajor;
pub const apiVersionMinor = types.apiVersionMinor;
pub const apiVersionPatch = types.apiVersionPatch;

/// Check if Vulkan is available on this system
pub fn isAvailable() bool {
    return BaseLoader.isAvailable();
}

test {
    @import("std").testing.refAllDecls(@This());
}
