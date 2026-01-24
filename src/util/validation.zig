// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! Validation Utilities
//!
//! This module provides compile-time and runtime validation for Vulkan operations.
//! Validation is enabled in debug builds and has zero overhead in release builds.
//!
//! ## Features
//!
//! - SPIR-V shader validation
//! - Buffer size limit validation
//! - Descriptor bounds checking
//! - Device limit verification
//!
//! ## Usage
//!
//! ```zig
//! const validation = @import("util/validation.zig");
//!
//! // Validate SPIR-V
//! const spirv_result = validation.validateSpirv(spirv_code);
//! if (spirv_result != .valid) {
//!     return error.InvalidSpirv;
//! }
//!
//! // Validate buffer size
//! try validation.validateBufferSize(size, limits, .storage);
//! ```

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../vk/types.zig");
const err = @import("../vk/error.zig");

/// Whether debug validation is enabled
pub const validation_enabled = builtin.mode == .Debug;

// =============================================================================
// SPIR-V Validation
// =============================================================================

/// SPIR-V validation result
pub const SpirvValidationResult = enum {
    /// SPIR-V is valid
    valid,
    /// Data is too short
    too_short,
    /// Invalid magic number
    invalid_magic,
    /// Invalid version
    invalid_version,
    /// Invalid bound (ID bound too large or zero)
    invalid_bound,
    /// Invalid schema (must be 0)
    invalid_schema,
    /// Code size not aligned to 4 bytes
    not_aligned,
};

/// SPIR-V header constants
pub const spirv = struct {
    /// SPIR-V magic number
    pub const MAGIC: u32 = 0x07230203;
    /// SPIR-V magic number (big-endian)
    pub const MAGIC_REV: u32 = 0x03022307;
    /// Minimum header size in words
    pub const MIN_HEADER_SIZE: usize = 5;
    /// Maximum reasonable bound (spec limit is 4M)
    pub const MAX_BOUND: u32 = 4 * 1024 * 1024;
    /// Current SPIR-V version (1.6)
    pub const VERSION_1_6: u32 = 0x00010600;
    /// Minimum supported version (1.0)
    pub const VERSION_1_0: u32 = 0x00010000;

    // Common SPIR-V opcodes for basic validation
    pub const OpNop: u16 = 0;
    pub const OpSource: u16 = 3;
    pub const OpName: u16 = 5;
    pub const OpMemberName: u16 = 6;
    pub const OpExtInstImport: u16 = 11;
    pub const OpMemoryModel: u16 = 14;
    pub const OpEntryPoint: u16 = 15;
    pub const OpExecutionMode: u16 = 16;
    pub const OpCapability: u16 = 17;
    pub const OpTypeVoid: u16 = 19;
    pub const OpTypeFunction: u16 = 33;
    pub const OpFunction: u16 = 54;
    pub const OpFunctionEnd: u16 = 56;
    pub const OpVariable: u16 = 59;
    pub const OpDecorate: u16 = 71;
    pub const OpMemberDecorate: u16 = 72;
};

/// Detailed SPIR-V validation information
pub const SpirvInfo = struct {
    /// Validation result
    result: SpirvValidationResult,
    /// SPIR-V version (major.minor)
    version_major: u8 = 0,
    version_minor: u8 = 0,
    /// Generator ID
    generator: u32 = 0,
    /// ID bound
    bound: u32 = 0,
    /// Number of instructions (approximate)
    instruction_count: u32 = 0,
    /// Whether entry point was found
    has_entry_point: bool = false,
    /// Whether memory model was specified
    has_memory_model: bool = false,

    pub fn isValid(self: SpirvInfo) bool {
        return self.result == .valid;
    }

    pub fn getVersionString(self: SpirvInfo) []const u8 {
        return switch (self.version_major) {
            1 => switch (self.version_minor) {
                0 => "1.0",
                1 => "1.1",
                2 => "1.2",
                3 => "1.3",
                4 => "1.4",
                5 => "1.5",
                6 => "1.6",
                else => "1.x",
            },
            else => "unknown",
        };
    }
};

/// Validate SPIR-V bytecode (basic check)
pub fn validateSpirv(spirv_code: []const u32) SpirvValidationResult {
    const info = validateSpirvDetailed(spirv_code);
    return info.result;
}

/// Validate SPIR-V bytecode with detailed information
pub fn validateSpirvDetailed(spirv_code: []const u32) SpirvInfo {
    var info = SpirvInfo{ .result = .valid };

    // Check minimum size
    if (spirv_code.len < spirv.MIN_HEADER_SIZE) {
        info.result = .too_short;
        return info;
    }

    // Check magic number
    const magic = spirv_code[0];
    if (magic != spirv.MAGIC) {
        if (magic == spirv.MAGIC_REV) {
            // Big-endian SPIR-V not supported
            info.result = .invalid_magic;
        } else {
            info.result = .invalid_magic;
        }
        return info;
    }

    // Parse version (word 1)
    // SPIR-V version format: 0 | Major (8 bits) | Minor (8 bits) | 0
    // The & 0xFF mask ensures the value fits in u8, so @truncate is safe
    const version = spirv_code[1];
    info.version_major = @truncate((version >> 16) & 0xFF);
    info.version_minor = @truncate((version >> 8) & 0xFF);

    // Validate version range
    if (version < spirv.VERSION_1_0 or version > spirv.VERSION_1_6 + 0x100) {
        info.result = .invalid_version;
        return info;
    }

    // Parse generator (word 2)
    info.generator = spirv_code[2];

    // Parse and validate bound (word 3)
    info.bound = spirv_code[3];
    if (info.bound == 0 or info.bound > spirv.MAX_BOUND) {
        info.result = .invalid_bound;
        return info;
    }

    // Validate schema (word 4) - must be 0
    const schema = spirv_code[4];
    if (schema != 0) {
        info.result = .invalid_schema;
        return info;
    }

    // Optional: scan instructions for basic validity (debug builds only)
    if (validation_enabled) {
        var offset: usize = 5;
        while (offset < spirv_code.len) {
            const instruction = spirv_code[offset];
            const word_count = instruction >> 16;
            const opcode: u16 = @truncate(instruction & 0xFFFF);

            // Word count must be at least 1
            if (word_count == 0) break;

            // Track key instructions
            if (opcode == spirv.OpEntryPoint) {
                info.has_entry_point = true;
            } else if (opcode == spirv.OpMemoryModel) {
                info.has_memory_model = true;
            }

            info.instruction_count += 1;
            offset += word_count;
        }
    }

    return info;
}

/// Validate SPIR-V from raw bytes
/// Note: If bytes are not 4-byte aligned in memory, this copies to a temporary aligned buffer.
pub fn validateSpirvBytes(bytes: []const u8) SpirvValidationResult {
    if (bytes.len % 4 != 0) {
        return .not_aligned;
    }

    // Check if bytes are properly aligned for u32 access
    if (@intFromPtr(bytes.ptr) % 4 != 0) {
        // Not aligned - need to copy to aligned buffer for validation
        // Use stack buffer for small shaders, otherwise report alignment issue
        if (bytes.len <= 4096) {
            var aligned_buf: [1024]u32 align(4) = undefined;
            const aligned_bytes = std.mem.sliceAsBytes(&aligned_buf);
            @memcpy(aligned_bytes[0..bytes.len], bytes);
            const spirv_code = aligned_buf[0 .. bytes.len / 4];
            return validateSpirv(spirv_code);
        } else {
            // Large unaligned buffer - report as alignment issue
            return .not_aligned;
        }
    }

    const spirv_code: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, bytes));
    return validateSpirv(spirv_code);
}

// =============================================================================
// Buffer Size Validation
// =============================================================================

/// Buffer type for limit validation
pub const BufferType = enum {
    storage,
    uniform,
    vertex,
    index,
    indirect,
    staging,
};

/// Validate buffer size against device limits
pub fn validateBufferSize(
    size: u64,
    limits: types.PhysicalDeviceLimits,
    buffer_type: BufferType,
) err.Error!void {
    if (size == 0) {
        return err.Error.InvalidArgument;
    }

    const limit: u64 = switch (buffer_type) {
        .storage => limits.max_storage_buffer_range,
        .uniform => limits.max_uniform_buffer_range,
        .vertex, .index, .indirect, .staging => std.math.maxInt(u32), // No specific limit
    };

    if (size > limit) {
        return err.Error.InvalidArgument;
    }
}

/// Check if allocation would exceed memory allocation count
pub fn validateAllocationCount(
    current_count: u64,
    limits: types.PhysicalDeviceLimits,
) err.Error!void {
    if (current_count >= limits.max_memory_allocation_count) {
        return err.Error.TooManyObjects;
    }
}

// =============================================================================
// Descriptor Validation
// =============================================================================

/// Descriptor set binding info for validation
pub const DescriptorBindingInfo = struct {
    binding: u32,
    descriptor_type: types.DescriptorType,
    buffer_size: u64,
    buffer_offset: u64 = 0,
};

/// Validate descriptor bindings against layout
pub fn validateDescriptorBindings(
    bindings: []const DescriptorBindingInfo,
    layout_bindings: []const struct { binding: u32, descriptor_type: types.DescriptorType, count: u32 },
    limits: types.PhysicalDeviceLimits,
) err.Error!void {
    // In release builds, skip validation for performance
    if (!validation_enabled) return;

    for (bindings) |b| {
        // Find matching layout binding
        var found = false;
        for (layout_bindings) |lb| {
            if (lb.binding == b.binding) {
                found = true;

                // Validate descriptor type matches
                if (lb.descriptor_type != b.descriptor_type) {
                    return err.Error.InvalidArgument;
                }

                // Validate buffer size for buffer descriptors
                switch (b.descriptor_type) {
                    .storage_buffer, .storage_buffer_dynamic => {
                        if (b.buffer_size > limits.max_storage_buffer_range) {
                            return err.Error.InvalidArgument;
                        }
                    },
                    .uniform_buffer, .uniform_buffer_dynamic => {
                        if (b.buffer_size > limits.max_uniform_buffer_range) {
                            return err.Error.InvalidArgument;
                        }
                    },
                    else => {},
                }
                break;
            }
        }

        if (!found) {
            return err.Error.InvalidArgument;
        }
    }
}

/// Validate descriptor pool capacity
pub fn validateDescriptorPoolCapacity(
    requested_sets: u32,
    max_sets: u32,
    allocated_sets: u32,
) err.Error!void {
    if (allocated_sets + requested_sets > max_sets) {
        return err.Error.OutOfPoolMemory;
    }
}

// =============================================================================
// Device Limits Helper
// =============================================================================

/// Get human-readable description of a device limit
pub fn getLimitDescription(limit_name: []const u8) []const u8 {
    const limits_info = std.StaticStringMap([]const u8).initComptime(.{
        .{ "max_storage_buffer_range", "Maximum size of a storage buffer binding" },
        .{ "max_uniform_buffer_range", "Maximum size of a uniform buffer binding" },
        .{ "max_push_constants_size", "Maximum size of push constant data" },
        .{ "max_memory_allocation_count", "Maximum number of device memory allocations" },
        .{ "max_bound_descriptor_sets", "Maximum number of descriptor sets bound simultaneously" },
        .{ "max_compute_work_group_count", "Maximum compute dispatch dimensions" },
        .{ "max_compute_work_group_size", "Maximum local work group size" },
        .{ "max_compute_shared_memory_size", "Maximum shared memory per work group" },
    });
    return limits_info.get(limit_name) orelse "Unknown limit";
}

// =============================================================================
// Tests
// =============================================================================

test "validateSpirv with valid SPIR-V" {
    const valid_spirv = [_]u32{
        spirv.MAGIC, // Magic
        0x00010000, // Version 1.0
        0, // Generator
        10, // Bound
        0, // Schema
    };
    try std.testing.expectEqual(SpirvValidationResult.valid, validateSpirv(&valid_spirv));
}

test "validateSpirv with invalid magic" {
    const invalid_spirv = [_]u32{ 0x12345678, 0x00010000, 0, 10, 0 };
    try std.testing.expectEqual(SpirvValidationResult.invalid_magic, validateSpirv(&invalid_spirv));
}

test "validateSpirv too short" {
    const short_spirv = [_]u32{ spirv.MAGIC, 0x00010000, 0 };
    try std.testing.expectEqual(SpirvValidationResult.too_short, validateSpirv(&short_spirv));
}

test "validateSpirv invalid bound" {
    const invalid_bound = [_]u32{ spirv.MAGIC, 0x00010000, 0, 0, 0 };
    try std.testing.expectEqual(SpirvValidationResult.invalid_bound, validateSpirv(&invalid_bound));
}

test "validateSpirv invalid schema" {
    const invalid_schema = [_]u32{ spirv.MAGIC, 0x00010000, 0, 10, 1 };
    try std.testing.expectEqual(SpirvValidationResult.invalid_schema, validateSpirv(&invalid_schema));
}

test "validateSpirvDetailed extracts version info" {
    const valid_spirv = [_]u32{ spirv.MAGIC, 0x00010300, 0x0001_0001, 100, 0 };
    const info = validateSpirvDetailed(&valid_spirv);
    try std.testing.expect(info.isValid());
    try std.testing.expectEqual(@as(u8, 1), info.version_major);
    try std.testing.expectEqual(@as(u8, 3), info.version_minor);
    try std.testing.expectEqual(@as(u32, 100), info.bound);
}

test "validateSpirvBytes with unaligned data" {
    const bytes = [_]u8{ 0, 1, 2 };
    try std.testing.expectEqual(SpirvValidationResult.not_aligned, validateSpirvBytes(&bytes));
}
