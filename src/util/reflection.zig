// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! SPIR-V Shader Reflection
//!
//! Parses SPIR-V bytecode to extract descriptor bindings, push constants,
//! and other metadata for automatic descriptor set layout generation.
//!
//! ## Features
//!
//! - Automatic descriptor binding extraction
//! - Push constant range detection
//! - Workgroup size extraction from compute shaders
//! - Input/output variable analysis
//!
//! ## Example
//!
//! ```zig
//! const vulkan = @import("vulkan");
//!
//! // Reflect a compute shader
//! const spirv_code = @embedFile("shader.spv");
//! var reflection = try vulkan.ShaderReflection.init(allocator, spirv_code);
//! defer reflection.deinit();
//!
//! // Get descriptor bindings
//! for (reflection.getBindings()) |binding| {
//!     std.debug.print("Set {}, Binding {}: {s}\n", .{
//!         binding.set, binding.binding, @tagName(binding.descriptor_type),
//!     });
//! }
//!
//! // Create descriptor layout automatically
//! const layout = try reflection.createDescriptorSetLayout(&ctx, 0);
//! defer layout.deinit();
//! ```

const std = @import("std");
const types = @import("../vk/types.zig");
const err = @import("../vk/error.zig");

// =============================================================================
// SPIR-V Constants
// =============================================================================

const SPIRV_MAGIC: u32 = 0x07230203;

// SPIR-V OpCodes we care about
const OpCode = enum(u16) {
    OpName = 5,
    OpMemberName = 6,
    OpEntryPoint = 15,
    OpExecutionMode = 16,
    OpTypeVoid = 19,
    OpTypeBool = 20,
    OpTypeInt = 21,
    OpTypeFloat = 22,
    OpTypeVector = 23,
    OpTypeMatrix = 24,
    OpTypeImage = 25,
    OpTypeSampler = 26,
    OpTypeSampledImage = 27,
    OpTypeArray = 28,
    OpTypeRuntimeArray = 29,
    OpTypeStruct = 30,
    OpTypePointer = 32,
    OpConstant = 43,
    OpSpecConstantTrue = 48,
    OpSpecConstantFalse = 49,
    OpSpecConstant = 50,
    OpSpecConstantOp = 52,
    OpVariable = 59,
    OpDecorate = 71,
    OpMemberDecorate = 72,
    _,
};

// SPIR-V Decorations
const Decoration = enum(u32) {
    Block = 2,
    BufferBlock = 3,
    RowMajor = 4,
    ColMajor = 5,
    ArrayStride = 6,
    MatrixStride = 7,
    BuiltIn = 11,
    NonWritable = 24,
    NonReadable = 25,
    Location = 30,
    Binding = 33,
    DescriptorSet = 34,
    Offset = 35,
    SpecId = 1, // For specialization constants
    InputAttachmentIndex = 43,
    _,
};

// SPIR-V Storage Classes
const StorageClass = enum(u32) {
    UniformConstant = 0,
    Input = 1,
    Uniform = 2,
    Output = 3,
    Workgroup = 4,
    CrossWorkgroup = 5,
    Private = 6,
    Function = 7,
    Generic = 8,
    PushConstant = 9,
    AtomicCounter = 10,
    Image = 11,
    StorageBuffer = 12,
    _,
};

// SPIR-V Execution Modes
const ExecutionMode = enum(u32) {
    LocalSize = 17,
    LocalSizeHint = 18,
    LocalSizeId = 38,
    _,
};

// SPIR-V Execution Models
const ExecutionModel = enum(u32) {
    Vertex = 0,
    TessellationControl = 1,
    TessellationEvaluation = 2,
    Geometry = 3,
    Fragment = 4,
    GLCompute = 5,
    Kernel = 6,
    _,
};

// SPIR-V Dim for images
const Dim = enum(u32) {
    Dim1D = 0,
    Dim2D = 1,
    Dim3D = 2,
    Cube = 3,
    Rect = 4,
    Buffer = 5,
    SubpassData = 6,
    _,
};

// =============================================================================
// Reflection Types
// =============================================================================

/// Descriptor binding information extracted from SPIR-V
pub const DescriptorBinding = struct {
    /// Descriptor set number
    set: u32,
    /// Binding number within the set
    binding: u32,
    /// Type of descriptor
    descriptor_type: types.DescriptorType,
    /// Number of descriptors (for arrays)
    count: u32 = 1,
    /// Shader stages that use this binding
    stage_flags: types.ShaderStageFlags = .{ .compute = true },
    /// Name of the variable (if available)
    name: ?[]const u8 = null,
    /// Whether this is a readonly resource
    readonly: bool = false,
    /// Whether this is a writeonly resource
    writeonly: bool = false,
    /// SPIR-V type ID (for advanced analysis)
    type_id: u32 = 0,
};

/// Push constant range information
pub const PushConstantInfo = struct {
    /// Offset in bytes
    offset: u32,
    /// Size in bytes
    size: u32,
    /// Stage flags
    stage_flags: types.ShaderStageFlags = .{ .compute = true },
    /// Name of the push constant block (if available)
    name: ?[]const u8 = null,
};

/// Workgroup size for compute shaders
pub const WorkgroupSize = struct {
    x: u32 = 1,
    y: u32 = 1,
    z: u32 = 1,

    pub fn total(self: WorkgroupSize) u32 {
        return self.x * self.y * self.z;
    }
};

/// Entry point information
pub const EntryPoint = struct {
    name: []const u8,
    execution_model: ExecutionModel,
    workgroup_size: ?WorkgroupSize = null,
};

/// Complete shader reflection data
pub const ShaderReflection = struct {
    allocator: std.mem.Allocator,
    /// All descriptor bindings
    bindings: std.ArrayListUnmanaged(DescriptorBinding),
    /// Push constant ranges
    push_constants: std.ArrayListUnmanaged(PushConstantInfo),
    /// Entry points
    entry_points: std.ArrayListUnmanaged(EntryPoint),
    /// Workgroup size (for compute shaders)
    workgroup_size: WorkgroupSize = .{},
    /// SPIR-V version
    spirv_version: u32 = 0,
    /// Bound (max ID + 1)
    bound: u32 = 0,

    // Internal parsing state
    names: std.AutoHashMapUnmanaged(u32, []const u8),
    decorations: std.AutoHashMapUnmanaged(u64, u32), // key = id << 32 | decoration
    member_decorations: std.AutoHashMapUnmanaged(u64, u32), // key = struct_id << 48 | member_idx << 32 | decoration
    type_info: std.AutoHashMapUnmanaged(u32, TypeInfo),
    variables: std.ArrayListUnmanaged(VariableInfo),
    constants: std.AutoHashMapUnmanaged(u32, u64), // constant ID -> value
    spec_constants: std.ArrayListUnmanaged(SpecConstantInfo),

    const Self = @This();

    /// Encode member decoration key as u64
    fn memberDecoKey(struct_id: u32, member_idx: u32, decoration: Decoration) u64 {
        return (@as(u64, struct_id) << 48) | (@as(u64, member_idx) << 32) | @as(u64, @intFromEnum(decoration));
    }

    const TypeInfo = struct {
        opcode: OpCode,
        storage_class: ?StorageClass = null,
        element_type: u32 = 0,
        dim: ?Dim = null,
        array_size: u32 = 1,
        is_sampled: bool = false,
        is_storage: bool = false,
        /// For scalar types (OpTypeInt, OpTypeFloat): bit width
        bit_width: u32 = 0,
        /// For signed integers
        is_signed: bool = false,
        /// For struct types - member type IDs (heap allocated)
        member_types: []const u32 = &.{},
        /// For vectors/matrices: component count
        component_count: u32 = 0,
        /// For matrices: column count
        column_count: u32 = 0,
    };

    /// Specialization constant info
    pub const SpecConstantInfo = struct {
        /// Constant ID (from SpecId decoration)
        constant_id: u32,
        /// Result ID in SPIR-V
        result_id: u32,
        /// Type ID
        type_id: u32,
        /// Default value (for scalar types)
        default_value: u64 = 0,
        /// Optional name
        name: ?[]const u8 = null,
    };

    const VariableInfo = struct {
        id: u32,
        type_id: u32,
        storage_class: StorageClass,
    };

    /// Initialize shader reflection from SPIR-V bytecode
    ///
    /// SPIR-V header format:
    /// - Word 0: Magic number (0x07230203)
    /// - Word 1: Version (major.minor.0.0)
    /// - Word 2: Generator magic number
    /// - Word 3: Bound (all IDs must be < bound)
    /// - Word 4: Schema (reserved, must be 0)
    /// - Word 5+: Instructions
    pub fn init(allocator: std.mem.Allocator, spirv: []const u8) !Self {
        // Minimum SPIR-V: 5-word header (20 bytes)
        if (spirv.len < 20) {
            std.log.warn("ShaderReflection: SPIR-V bytecode too small ({d} bytes, minimum 20)", .{spirv.len});
            return err.Error.InvalidSpirv;
        }

        // Must be word-aligned
        if (spirv.len % 4 != 0) {
            std.log.warn("ShaderReflection: SPIR-V bytecode not word-aligned ({d} bytes)", .{spirv.len});
            return err.Error.InvalidSpirv;
        }

        const words: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, spirv));
        if (words.len < 5) {
            std.log.warn("ShaderReflection: SPIR-V word count too small ({d} words, minimum 5)", .{words.len});
            return err.Error.InvalidSpirv;
        }
        if (words[0] != SPIRV_MAGIC) {
            std.log.warn("ShaderReflection: invalid SPIR-V magic number (got 0x{X:0>8}, expected 0x{X:0>8})", .{ words[0], SPIRV_MAGIC });
            return err.Error.InvalidSpirv;
        }

        var self = Self{
            .allocator = allocator,
            .bindings = .{},
            .push_constants = .{},
            .entry_points = .{},
            .names = .{},
            .decorations = .{},
            .member_decorations = .{},
            .type_info = .{},
            .variables = .{},
            .constants = .{},
            .spec_constants = .{},
            .spirv_version = words[1],
            .bound = words[3],
        };

        errdefer self.deinit();

        // Parse SPIR-V instructions
        try self.parseInstructions(words[5..]);

        // Build descriptor bindings from parsed data
        try self.buildBindings();

        return self;
    }

    /// Deinitialize and free all resources
    pub fn deinit(self: *Self) void {
        // Free allocated names
        var name_iter = self.names.valueIterator();
        while (name_iter.next()) |name| {
            self.allocator.free(name.*);
        }
        self.names.deinit(self.allocator);

        // Free entry point names
        for (self.entry_points.items) |ep| {
            self.allocator.free(ep.name);
        }
        self.entry_points.deinit(self.allocator);

        // Free binding names
        for (self.bindings.items) |*binding| {
            if (binding.name) |name| {
                self.allocator.free(name);
            }
        }
        self.bindings.deinit(self.allocator);

        // Free push constant names
        for (self.push_constants.items) |*pc| {
            if (pc.name) |name| {
                self.allocator.free(name);
            }
        }
        self.push_constants.deinit(self.allocator);

        // Free struct member type arrays
        var type_iter = self.type_info.valueIterator();
        while (type_iter.next()) |type_info| {
            if (type_info.member_types.len > 0) {
                self.allocator.free(type_info.member_types);
            }
        }
        self.type_info.deinit(self.allocator);

        // Free spec constant names
        for (self.spec_constants.items) |*sc| {
            if (sc.name) |name| {
                self.allocator.free(name);
            }
        }
        self.spec_constants.deinit(self.allocator);

        self.decorations.deinit(self.allocator);
        self.member_decorations.deinit(self.allocator);
        self.constants.deinit(self.allocator);
        self.variables.deinit(self.allocator);
        self.* = undefined;
    }

    /// Parse SPIR-V instructions
    fn parseInstructions(self: *Self, words: []const u32) !void {
        var i: usize = 0;
        while (i < words.len) {
            const instruction = words[i];
            const word_count = instruction >> 16;
            const opcode: OpCode = @enumFromInt(@as(u16, @truncate(instruction)));

            if (word_count == 0 or i + word_count > words.len) break;

            const operands = words[i + 1 .. i + word_count];

            switch (opcode) {
                .OpName => try self.parseOpName(operands),
                .OpDecorate => try self.parseOpDecorate(operands),
                .OpMemberDecorate => try self.parseOpMemberDecorate(operands),
                .OpEntryPoint => try self.parseOpEntryPoint(operands),
                .OpExecutionMode => try self.parseOpExecutionMode(operands),
                .OpVariable => try self.parseOpVariable(operands),
                .OpTypeInt => try self.parseOpTypeInt(operands),
                .OpTypeFloat => try self.parseOpTypeFloat(operands),
                .OpTypeVector => try self.parseOpTypeVector(operands),
                .OpTypeMatrix => try self.parseOpTypeMatrix(operands),
                .OpTypePointer => try self.parseOpTypePointer(operands),
                .OpTypeStruct => try self.parseOpTypeStruct(operands),
                .OpTypeImage => try self.parseOpTypeImage(operands),
                .OpTypeSampler => try self.parseOpTypeSampler(operands),
                .OpTypeSampledImage => try self.parseOpTypeSampledImage(operands),
                .OpTypeArray => try self.parseOpTypeArray(operands),
                .OpTypeRuntimeArray => try self.parseOpTypeRuntimeArray(operands),
                .OpConstant => try self.parseOpConstant(operands),
                .OpSpecConstant, .OpSpecConstantTrue, .OpSpecConstantFalse => try self.parseOpSpecConstant(operands, opcode),
                else => {},
            }

            i += word_count;
        }
    }

    fn parseOpName(self: *Self, operands: []const u32) !void {
        if (operands.len < 2) return;
        const id = operands[0];
        const name_bytes = std.mem.sliceAsBytes(operands[1..]);
        const name_end = std.mem.indexOfScalar(u8, name_bytes, 0) orelse name_bytes.len;
        const name = try self.allocator.dupe(u8, name_bytes[0..name_end]);
        try self.names.put(self.allocator, id, name);
    }

    fn parseOpDecorate(self: *Self, operands: []const u32) !void {
        if (operands.len < 2) return;
        const id = operands[0];
        const decoration: Decoration = @enumFromInt(operands[1]);
        const value = if (operands.len > 2) operands[2] else 0;

        const key = (@as(u64, id) << 32) | @as(u64, @intFromEnum(decoration));
        try self.decorations.put(self.allocator, key, value);
    }

    fn parseOpMemberDecorate(self: *Self, operands: []const u32) !void {
        if (operands.len < 3) return;
        const struct_id = operands[0];
        const member_idx = operands[1];
        const decoration: Decoration = @enumFromInt(operands[2]);
        const value = if (operands.len > 3) operands[3] else 0;

        const key = memberDecoKey(struct_id, member_idx, decoration);
        try self.member_decorations.put(self.allocator, key, value);
    }

    fn parseOpEntryPoint(self: *Self, operands: []const u32) !void {
        if (operands.len < 3) return;
        const execution_model: ExecutionModel = @enumFromInt(operands[0]);
        // operands[1] is the function ID
        const name_bytes = std.mem.sliceAsBytes(operands[2..]);
        const name_end = std.mem.indexOfScalar(u8, name_bytes, 0) orelse name_bytes.len;
        const name = try self.allocator.dupe(u8, name_bytes[0..name_end]);

        try self.entry_points.append(self.allocator, .{
            .name = name,
            .execution_model = execution_model,
        });
    }

    fn parseOpExecutionMode(self: *Self, operands: []const u32) !void {
        if (operands.len < 2) return;
        // operands[0] is the entry point ID
        const mode: ExecutionMode = @enumFromInt(operands[1]);

        if (mode == .LocalSize and operands.len >= 5) {
            self.workgroup_size = .{
                .x = operands[2],
                .y = operands[3],
                .z = operands[4],
            };
        }
    }

    fn parseOpVariable(self: *Self, operands: []const u32) !void {
        if (operands.len < 3) return;
        const type_id = operands[0];
        const result_id = operands[1];
        const storage_class: StorageClass = @enumFromInt(operands[2]);

        try self.variables.append(self.allocator, .{
            .id = result_id,
            .type_id = type_id,
            .storage_class = storage_class,
        });
    }

    fn parseOpTypeInt(self: *Self, operands: []const u32) !void {
        if (operands.len < 3) return;
        const result_id = operands[0];
        const bit_width = operands[1];
        const signedness = operands[2];

        try self.type_info.put(self.allocator, result_id, .{
            .opcode = .OpTypeInt,
            .bit_width = bit_width,
            .is_signed = signedness != 0,
        });
    }

    fn parseOpTypeFloat(self: *Self, operands: []const u32) !void {
        if (operands.len < 2) return;
        const result_id = operands[0];
        const bit_width = operands[1];

        try self.type_info.put(self.allocator, result_id, .{
            .opcode = .OpTypeFloat,
            .bit_width = bit_width,
        });
    }

    fn parseOpTypeVector(self: *Self, operands: []const u32) !void {
        if (operands.len < 3) return;
        const result_id = operands[0];
        const component_type = operands[1];
        const component_count = operands[2];

        try self.type_info.put(self.allocator, result_id, .{
            .opcode = .OpTypeVector,
            .element_type = component_type,
            .component_count = component_count,
        });
    }

    fn parseOpTypeMatrix(self: *Self, operands: []const u32) !void {
        if (operands.len < 3) return;
        const result_id = operands[0];
        const column_type = operands[1]; // This is a vector type
        const column_count = operands[2];

        try self.type_info.put(self.allocator, result_id, .{
            .opcode = .OpTypeMatrix,
            .element_type = column_type,
            .column_count = column_count,
        });
    }

    fn parseOpTypePointer(self: *Self, operands: []const u32) !void {
        if (operands.len < 3) return;
        const result_id = operands[0];
        const storage_class: StorageClass = @enumFromInt(operands[1]);
        const element_type = operands[2];

        try self.type_info.put(self.allocator, result_id, .{
            .opcode = .OpTypePointer,
            .storage_class = storage_class,
            .element_type = element_type,
        });
    }

    fn parseOpTypeStruct(self: *Self, operands: []const u32) !void {
        if (operands.len < 1) return;
        const result_id = operands[0];

        // Member type IDs start at operand[1]
        const member_types = if (operands.len > 1)
            try self.allocator.dupe(u32, operands[1..])
        else
            &[_]u32{};

        try self.type_info.put(self.allocator, result_id, .{
            .opcode = .OpTypeStruct,
            .member_types = member_types,
        });
    }

    fn parseOpTypeImage(self: *Self, operands: []const u32) !void {
        if (operands.len < 2) return;
        const result_id = operands[0];
        const dim: Dim = if (operands.len > 2) @enumFromInt(operands[2]) else .Dim2D;
        // operands[6] is Sampled: 0 = runtime, 1 = sampled, 2 = storage
        const sampled = if (operands.len > 6) operands[6] else 1;

        try self.type_info.put(self.allocator, result_id, .{
            .opcode = .OpTypeImage,
            .dim = dim,
            .is_sampled = sampled == 1,
            .is_storage = sampled == 2,
        });
    }

    fn parseOpTypeSampler(self: *Self, operands: []const u32) !void {
        if (operands.len < 1) return;
        const result_id = operands[0];

        try self.type_info.put(self.allocator, result_id, .{
            .opcode = .OpTypeSampler,
        });
    }

    fn parseOpTypeSampledImage(self: *Self, operands: []const u32) !void {
        if (operands.len < 2) return;
        const result_id = operands[0];
        const image_type = operands[1];

        try self.type_info.put(self.allocator, result_id, .{
            .opcode = .OpTypeSampledImage,
            .element_type = image_type,
        });
    }

    fn parseOpTypeArray(self: *Self, operands: []const u32) !void {
        if (operands.len < 3) return;
        const result_id = operands[0];
        const element_type = operands[1];
        // operands[2] is the length constant ID - we'd need to look it up

        try self.type_info.put(self.allocator, result_id, .{
            .opcode = .OpTypeArray,
            .element_type = element_type,
        });
    }

    fn parseOpTypeRuntimeArray(self: *Self, operands: []const u32) !void {
        if (operands.len < 2) return;
        const result_id = operands[0];
        const element_type = operands[1];

        try self.type_info.put(self.allocator, result_id, .{
            .opcode = .OpTypeRuntimeArray,
            .element_type = element_type,
        });
    }

    fn parseOpConstant(self: *Self, operands: []const u32) !void {
        if (operands.len < 3) return;
        // operands[0] = type ID
        const result_id = operands[1];
        // operands[2..] = literal value (may be multiple words for 64-bit)
        const value: u64 = if (operands.len > 3)
            (@as(u64, operands[3]) << 32) | @as(u64, operands[2])
        else
            @as(u64, operands[2]);

        try self.constants.put(self.allocator, result_id, value);
    }

    fn parseOpSpecConstant(self: *Self, operands: []const u32, opcode: OpCode) !void {
        if (operands.len < 2) return;
        const type_id = operands[0];
        const result_id = operands[1];

        // Get default value
        const default_value: u64 = switch (opcode) {
            .OpSpecConstantTrue => 1,
            .OpSpecConstantFalse => 0,
            .OpSpecConstant => if (operands.len > 2)
                (if (operands.len > 3)
                    (@as(u64, operands[3]) << 32) | @as(u64, operands[2])
                else
                    @as(u64, operands[2]))
            else
                0,
            else => 0,
        };

        // Get SpecId decoration if present
        const spec_id_key = (@as(u64, result_id) << 32) | @as(u64, @intFromEnum(Decoration.SpecId));
        const constant_id = self.decorations.get(spec_id_key) orelse return;

        // Get name if available
        const name = if (self.names.get(result_id)) |n|
            try self.allocator.dupe(u8, n)
        else
            null;

        try self.spec_constants.append(self.allocator, .{
            .constant_id = constant_id,
            .result_id = result_id,
            .type_id = type_id,
            .default_value = default_value,
            .name = name,
        });
    }

    /// Calculate the size of a SPIR-V type in bytes
    /// Returns null if the size cannot be determined
    fn calculateTypeSize(self: *Self, type_id: u32) ?u32 {
        const type_info = self.type_info.get(type_id) orelse return null;

        return switch (type_info.opcode) {
            .OpTypeInt, .OpTypeFloat => type_info.bit_width / 8,
            .OpTypeBool => 4, // GLSL bool is 32-bit in Vulkan
            .OpTypeVector => blk: {
                const component_size = self.calculateTypeSize(type_info.element_type) orelse return null;
                break :blk component_size * type_info.component_count;
            },
            .OpTypeMatrix => blk: {
                // Matrix is array of column vectors
                const column_size = self.calculateTypeSize(type_info.element_type) orelse return null;
                // Apply std140/std430 alignment rules: columns are aligned to vec4
                const aligned_column_size = (column_size + 15) & ~@as(u32, 15);
                break :blk aligned_column_size * type_info.column_count;
            },
            .OpTypeArray => blk: {
                const element_size = self.calculateTypeSize(type_info.element_type) orelse return null;
                // Array stride may be specified by decoration, otherwise use element size with alignment
                const array_stride = self.getArrayStride(type_id) orelse element_size;
                break :blk array_stride * type_info.array_size;
            },
            .OpTypeStruct => self.calculateStructSize(type_id),
            .OpTypePointer => 8, // 64-bit pointers in buffer device address
            else => null,
        };
    }

    /// Calculate struct size from member offsets or by summing member sizes
    fn calculateStructSize(self: *Self, struct_id: u32) ?u32 {
        const type_info = self.type_info.get(struct_id) orelse return null;
        if (type_info.opcode != .OpTypeStruct) return null;

        const member_types = type_info.member_types;
        if (member_types.len == 0) return 0;

        // Try to use explicit member offsets from OpMemberDecorate
        var max_offset_plus_size: u32 = 0;
        var has_offsets = false;

        for (member_types, 0..) |member_type_id, member_idx| {
            const offset_key = memberDecoKey(struct_id, @intCast(member_idx), .Offset);
            if (self.member_decorations.get(offset_key)) |offset| {
                has_offsets = true;
                const member_size = self.calculateTypeSize(member_type_id) orelse 4;
                const end = offset + member_size;
                if (end > max_offset_plus_size) {
                    max_offset_plus_size = end;
                }
            }
        }

        if (has_offsets) {
            // Round up to 16-byte alignment (std140 struct alignment)
            return (max_offset_plus_size + 15) & ~@as(u32, 15);
        }

        // Fall back to summing member sizes (less accurate but better than nothing)
        var total_size: u32 = 0;
        for (member_types) |member_type_id| {
            const member_size = self.calculateTypeSize(member_type_id) orelse 4;
            // Apply basic alignment (size up to alignment boundary)
            const alignment = @min(member_size, 16);
            total_size = (total_size + alignment - 1) & ~(alignment - 1);
            total_size += member_size;
        }

        // Round up to 16-byte alignment
        return (total_size + 15) & ~@as(u32, 15);
    }

    /// Get array stride decoration if present
    fn getArrayStride(self: *Self, type_id: u32) ?u32 {
        const key = (@as(u64, type_id) << 32) | @as(u64, @intFromEnum(Decoration.ArrayStride));
        return self.decorations.get(key);
    }

    /// Build descriptor bindings from parsed data
    fn buildBindings(self: *Self) !void {
        for (self.variables.items) |variable| {
            // Get binding and set decorations
            const binding_key = (@as(u64, variable.id) << 32) | @as(u64, @intFromEnum(Decoration.Binding));
            const set_key = (@as(u64, variable.id) << 32) | @as(u64, @intFromEnum(Decoration.DescriptorSet));

            const binding_num = self.decorations.get(binding_key) orelse continue;
            const set_num = self.decorations.get(set_key) orelse 0;

            // Determine descriptor type from storage class and type
            const descriptor_type = self.determineDescriptorType(variable) orelse continue;

            // Check for readonly/writeonly
            const readonly_key = (@as(u64, variable.id) << 32) | @as(u64, @intFromEnum(Decoration.NonWritable));
            const writeonly_key = (@as(u64, variable.id) << 32) | @as(u64, @intFromEnum(Decoration.NonReadable));

            // Get name if available
            const name = if (self.names.get(variable.id)) |n|
                try self.allocator.dupe(u8, n)
            else
                null;

            try self.bindings.append(self.allocator, .{
                .set = set_num,
                .binding = binding_num,
                .descriptor_type = descriptor_type,
                .name = name,
                .readonly = self.decorations.contains(readonly_key),
                .writeonly = self.decorations.contains(writeonly_key),
                .type_id = variable.type_id,
            });
        }

        // Sort bindings by set, then by binding
        std.mem.sort(DescriptorBinding, self.bindings.items, {}, struct {
            fn lessThan(_: void, a: DescriptorBinding, b: DescriptorBinding) bool {
                if (a.set != b.set) return a.set < b.set;
                return a.binding < b.binding;
            }
        }.lessThan);
    }

    fn determineDescriptorType(self: *Self, variable: VariableInfo) ?types.DescriptorType {
        // Get the pointer type
        const ptr_type = self.type_info.get(variable.type_id) orelse return null;

        switch (variable.storage_class) {
            .Uniform => {
                // Check if it's a Block (uniform buffer) or BufferBlock (storage buffer)
                const block_key = (@as(u64, ptr_type.element_type) << 32) | @as(u64, @intFromEnum(Decoration.Block));
                const buffer_block_key = (@as(u64, ptr_type.element_type) << 32) | @as(u64, @intFromEnum(Decoration.BufferBlock));

                if (self.decorations.contains(buffer_block_key)) {
                    return .storage_buffer;
                } else if (self.decorations.contains(block_key)) {
                    return .uniform_buffer;
                }
                return .uniform_buffer;
            },
            .StorageBuffer => return .storage_buffer,
            .UniformConstant => {
                // Could be sampler, image, or combined image sampler
                const element_type = self.type_info.get(ptr_type.element_type) orelse return null;

                return switch (element_type.opcode) {
                    .OpTypeSampler => .sampler,
                    .OpTypeSampledImage => .combined_image_sampler,
                    .OpTypeImage => if (element_type.is_storage) .storage_image else .sampled_image,
                    else => null,
                };
            },
            .PushConstant => {
                // Handle push constants separately
                const struct_type_id = ptr_type.element_type;
                const element_type = self.type_info.get(struct_type_id) orelse return null;
                if (element_type.opcode == .OpTypeStruct) {
                    // Calculate actual size from struct layout
                    const size = self.calculateStructSize(struct_type_id) orelse 128;

                    const name = if (self.names.get(variable.id)) |n|
                        self.allocator.dupe(u8, n) catch null
                    else
                        null;

                    self.push_constants.append(self.allocator, .{
                        .offset = 0,
                        .size = size,
                        .name = name,
                    }) catch |e| {
                        std.log.warn("ShaderReflection: failed to track push constant: {s} - push constants may be incomplete", .{
                            @errorName(e),
                        });
                    };
                }
                return null;
            },
            else => return null,
        }
    }

    // =========================================================================
    // Public API
    // =========================================================================

    /// Get all descriptor bindings
    pub fn getBindings(self: Self) []const DescriptorBinding {
        return self.bindings.items;
    }

    /// Get bindings for a specific descriptor set
    pub fn getBindingsForSet(self: Self, set: u32) []const DescriptorBinding {
        var start: usize = 0;
        var end: usize = 0;

        for (self.bindings.items, 0..) |binding, i| {
            if (binding.set == set) {
                if (start == end) start = i;
                end = i + 1;
            } else if (binding.set > set) {
                break;
            }
        }

        return self.bindings.items[start..end];
    }

    /// Get push constant information
    pub fn getPushConstants(self: Self) []const PushConstantInfo {
        return self.push_constants.items;
    }

    /// Get specialization constants
    pub fn getSpecConstants(self: Self) []const SpecConstantInfo {
        return self.spec_constants.items;
    }

    /// Get workgroup size for compute shaders
    pub fn getWorkgroupSize(self: Self) WorkgroupSize {
        return self.workgroup_size;
    }

    /// Get entry points
    pub fn getEntryPoints(self: Self) []const EntryPoint {
        return self.entry_points.items;
    }

    /// Get the number of unique descriptor sets used
    pub fn getDescriptorSetCount(self: Self) u32 {
        if (self.bindings.items.len == 0) return 0;

        var max_set: u32 = 0;
        for (self.bindings.items) |binding| {
            if (binding.set > max_set) max_set = binding.set;
        }
        return max_set + 1;
    }

    /// Check if this is a compute shader
    pub fn isCompute(self: Self) bool {
        for (self.entry_points.items) |ep| {
            if (ep.execution_model == .GLCompute or ep.execution_model == .Kernel) {
                return true;
            }
        }
        return false;
    }

    /// Create Vulkan descriptor set layout bindings for a specific set
    pub fn createLayoutBindings(
        self: Self,
        allocator: std.mem.Allocator,
        set: u32,
    ) ![]types.DescriptorSetLayoutBinding {
        const set_bindings = self.getBindingsForSet(set);
        if (set_bindings.len == 0) return &[_]types.DescriptorSetLayoutBinding{};

        var layout_bindings = try allocator.alloc(types.DescriptorSetLayoutBinding, set_bindings.len);
        errdefer allocator.free(layout_bindings);

        for (set_bindings, 0..) |binding, i| {
            layout_bindings[i] = .{
                .binding = binding.binding,
                .descriptor_type = binding.descriptor_type,
                .descriptor_count = binding.count,
                .stage_flags = binding.stage_flags,
                .p_immutable_samplers = null,
            };
        }

        return layout_bindings;
    }

    /// Print reflection summary for debugging
    pub fn printSummary(self: Self, writer: anytype) !void {
        try writer.print("SPIR-V Reflection Summary\n", .{});
        try writer.print("=========================\n", .{});
        try writer.print("Version: {}.{}\n", .{
            (self.spirv_version >> 16) & 0xFF,
            (self.spirv_version >> 8) & 0xFF,
        });

        if (self.entry_points.items.len > 0) {
            try writer.print("\nEntry Points:\n", .{});
            for (self.entry_points.items) |ep| {
                try writer.print("  - {s} ({s})\n", .{ ep.name, @tagName(ep.execution_model) });
            }
        }

        if (self.isCompute()) {
            try writer.print("\nWorkgroup Size: {}x{}x{}\n", .{
                self.workgroup_size.x,
                self.workgroup_size.y,
                self.workgroup_size.z,
            });
        }

        if (self.bindings.items.len > 0) {
            try writer.print("\nDescriptor Bindings:\n", .{});
            for (self.bindings.items) |binding| {
                try writer.print("  Set {}, Binding {}: {s}", .{
                    binding.set,
                    binding.binding,
                    @tagName(binding.descriptor_type),
                });
                if (binding.name) |name| {
                    try writer.print(" ({s})", .{name});
                }
                if (binding.readonly) try writer.print(" [readonly]", .{});
                if (binding.writeonly) try writer.print(" [writeonly]", .{});
                try writer.print("\n", .{});
            }
        }

        if (self.push_constants.items.len > 0) {
            try writer.print("\nPush Constants:\n", .{});
            for (self.push_constants.items) |pc| {
                try writer.print("  Offset: {}, Size: {} bytes", .{ pc.offset, pc.size });
                if (pc.name) |name| {
                    try writer.print(" ({s})", .{name});
                }
                try writer.print("\n", .{});
            }
        }
    }
};

/// Helper to create descriptor set layout from reflection
pub const ReflectedLayout = struct {
    reflection: *const ShaderReflection,
    set: u32,

    const Self = @This();

    /// Get the bindings for this set
    pub fn getBindings(self: Self) []const DescriptorBinding {
        return self.reflection.getBindingsForSet(self.set);
    }

    /// Get binding count
    pub fn bindingCount(self: Self) usize {
        return self.getBindings().len;
    }

    /// Check if a specific binding exists
    pub fn hasBinding(self: Self, binding: u32) bool {
        for (self.getBindings()) |b| {
            if (b.binding == binding) return true;
        }
        return false;
    }

    /// Get descriptor type for a binding
    pub fn getDescriptorType(self: Self, binding: u32) ?types.DescriptorType {
        for (self.getBindings()) |b| {
            if (b.binding == binding) return b.descriptor_type;
        }
        return null;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "ShaderReflection basic parsing" {
    // Minimal valid SPIR-V header (no actual instructions)
    const minimal_spirv = [_]u32{
        SPIRV_MAGIC, // Magic
        0x00010500, // Version 1.5
        0, // Generator
        1, // Bound
        0, // Reserved
    };

    const allocator = std.testing.allocator;
    var reflection = try ShaderReflection.init(allocator, std.mem.sliceAsBytes(&minimal_spirv));
    defer reflection.deinit();

    try std.testing.expectEqual(@as(u32, 0x00010500), reflection.spirv_version);
    try std.testing.expectEqual(@as(u32, 1), reflection.bound);
}

test "WorkgroupSize total calculation" {
    const wg = WorkgroupSize{ .x = 16, .y = 16, .z = 4 };
    try std.testing.expectEqual(@as(u32, 1024), wg.total());
}

test "DescriptorBinding comparison" {
    const b1 = DescriptorBinding{ .set = 0, .binding = 0, .descriptor_type = .storage_buffer };
    const b2 = DescriptorBinding{ .set = 0, .binding = 1, .descriptor_type = .uniform_buffer };
    const b3 = DescriptorBinding{ .set = 1, .binding = 0, .descriptor_type = .sampler };

    try std.testing.expect(b1.set == b2.set);
    try std.testing.expect(b1.binding < b2.binding);
    try std.testing.expect(b1.set < b3.set);
}
