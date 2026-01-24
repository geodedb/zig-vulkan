// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024-2025 CodePros

//! GLSL to SPIR-V Compilation Wrapper
//!
//! Provides utilities for compiling GLSL shaders to SPIR-V bytecode.
//! Supports both runtime compilation (via glslc/shaderc) and build-time compilation.
//!
//! ## Runtime Compilation
//!
//! ```zig
//! const glsl = @import("vulkan").glsl;
//!
//! // Compile GLSL source to SPIR-V
//! const spirv = try glsl.compile(allocator, .{
//!     .source = @embedFile("shader.comp"),
//!     .stage = .compute,
//!     .entry_point = "main",
//! });
//! defer allocator.free(spirv);
//!
//! // Use the compiled SPIR-V
//! var shader = try ShaderModule.initFromSpirv(&ctx, spirv);
//! ```
//!
//! ## Build-time Compilation
//!
//! For production, prefer build-time compilation using build.zig:
//!
//! ```zig
//! // In build.zig
//! const spirv = glsl.addCompileStep(b, "shaders/compute.glsl", .compute);
//! exe.addAnonymousModule("compute_shader", spirv);
//! ```
//!
//! ## Requirements
//!
//! Runtime compilation requires one of:
//! - glslc (from Vulkan SDK)
//! - glslangValidator (from Vulkan SDK)
//!
//! The compiler is searched in PATH and common installation locations.

const std = @import("std");
const builtin = @import("builtin");

/// SPIR-V magic number
const SPIRV_MAGIC: u32 = 0x07230203;

/// Validate SPIR-V bytecode
fn isValidSpirv(spirv: []const u32) bool {
    if (spirv.len < 5) return false;
    if (spirv[0] != SPIRV_MAGIC) return false;
    // Validate version (1.0 - 1.6)
    const version = spirv[1];
    const major = (version >> 16) & 0xFF;
    const minor = (version >> 8) & 0xFF;
    if (major != 1 or minor > 6) return false;
    // Check bound is non-zero
    if (spirv[3] == 0) return false;
    // Schema must be 0
    if (spirv[4] != 0) return false;
    return true;
}

/// Shader stage for compilation
pub const ShaderStage = enum {
    vertex,
    fragment,
    compute,
    geometry,
    tessellation_control,
    tessellation_evaluation,
    ray_generation,
    ray_intersection,
    ray_any_hit,
    ray_closest_hit,
    ray_miss,
    task,
    mesh,

    /// Get the glslc stage flag
    pub fn toGlslcFlag(self: ShaderStage) []const u8 {
        return switch (self) {
            .vertex => "-fshader-stage=vertex",
            .fragment => "-fshader-stage=fragment",
            .compute => "-fshader-stage=compute",
            .geometry => "-fshader-stage=geometry",
            .tessellation_control => "-fshader-stage=tesscontrol",
            .tessellation_evaluation => "-fshader-stage=tesseval",
            .ray_generation => "-fshader-stage=rgen",
            .ray_intersection => "-fshader-stage=rint",
            .ray_any_hit => "-fshader-stage=rahit",
            .ray_closest_hit => "-fshader-stage=rchit",
            .ray_miss => "-fshader-stage=rmiss",
            .task => "-fshader-stage=task",
            .mesh => "-fshader-stage=mesh",
        };
    }

    /// Get file extension for shader stage
    pub fn fileExtension(self: ShaderStage) []const u8 {
        return switch (self) {
            .vertex => ".vert",
            .fragment => ".frag",
            .compute => ".comp",
            .geometry => ".geom",
            .tessellation_control => ".tesc",
            .tessellation_evaluation => ".tese",
            .ray_generation => ".rgen",
            .ray_intersection => ".rint",
            .ray_any_hit => ".rahit",
            .ray_closest_hit => ".rchit",
            .ray_miss => ".rmiss",
            .task => ".task",
            .mesh => ".mesh",
        };
    }
};

/// GLSL version/profile
pub const GlslVersion = enum {
    @"450",
    @"460",

    pub fn toString(self: GlslVersion) []const u8 {
        return switch (self) {
            .@"450" => "450",
            .@"460" => "460",
        };
    }
};

/// Optimization level for shader compilation
pub const OptimizationLevel = enum {
    /// No optimization (debug)
    none,
    /// Size optimization
    size,
    /// Performance optimization
    performance,

    pub fn toGlslcFlag(self: OptimizationLevel) ?[]const u8 {
        return switch (self) {
            .none => "-O0",
            .size => "-Os",
            .performance => "-O",
        };
    }
};

/// Compilation options
pub const CompileOptions = struct {
    /// GLSL source code
    source: []const u8,
    /// Shader stage
    stage: ShaderStage,
    /// Entry point name
    entry_point: []const u8 = "main",
    /// GLSL version
    version: GlslVersion = .@"450",
    /// Optimization level
    optimization: OptimizationLevel = .performance,
    /// Generate debug info
    debug_info: bool = false,
    /// Target Vulkan version (e.g., "1.0", "1.1", "1.2", "1.3")
    target_vulkan: []const u8 = "1.0",
    /// Additional defines (name=value pairs)
    defines: []const Define = &.{},
    /// Include paths for #include directives
    include_paths: []const []const u8 = &.{},
};

/// Preprocessor define
pub const Define = struct {
    name: []const u8,
    value: ?[]const u8 = null,
};

/// Compilation result
pub const CompileResult = struct {
    /// Compiled SPIR-V bytecode (as u32 words)
    spirv: []const u32,
    /// Allocator used (for cleanup)
    allocator: std.mem.Allocator,

    pub fn deinit(self: *CompileResult) void {
        const bytes: []const u8 = @as([*]const u8, @ptrCast(self.spirv.ptr))[0 .. self.spirv.len * 4];
        self.allocator.free(@alignCast(bytes));
        self.* = undefined;
    }
};

/// Compilation error
pub const CompileError = error{
    /// No GLSL compiler found on system
    CompilerNotFound,
    /// Compilation failed (syntax error, etc.)
    CompilationFailed,
    /// Invalid shader source
    InvalidSource,
    /// Out of memory
    OutOfMemory,
    /// I/O error during compilation
    IoError,
    /// Process execution failed
    ProcessError,
};

/// Compiler backend
pub const CompilerBackend = enum {
    glslc,
    glslang_validator,

    pub fn executable(self: CompilerBackend) []const u8 {
        return switch (self) {
            .glslc => "glslc",
            .glslang_validator => "glslangValidator",
        };
    }
};

/// GLSL compiler wrapper
pub const Compiler = struct {
    allocator: std.mem.Allocator,
    backend: CompilerBackend,
    compiler_path: []const u8,

    const Self = @This();

    /// Find and initialize a GLSL compiler
    pub fn init(allocator: std.mem.Allocator) CompileError!Self {
        // Try to find glslc first (preferred)
        if (findCompiler(allocator, .glslc)) |path| {
            return Self{
                .allocator = allocator,
                .backend = .glslc,
                .compiler_path = path,
            };
        }

        // Fall back to glslangValidator
        if (findCompiler(allocator, .glslang_validator)) |path| {
            return Self{
                .allocator = allocator,
                .backend = .glslang_validator,
                .compiler_path = path,
            };
        }

        return CompileError.CompilerNotFound;
    }

    /// Initialize with a specific compiler path
    pub fn initWithPath(allocator: std.mem.Allocator, path: []const u8, backend: CompilerBackend) Self {
        return Self{
            .allocator = allocator,
            .backend = backend,
            .compiler_path = path,
        };
    }

    pub fn deinit(self: *Self) void {
        // compiler_path is statically defined, no need to free
        self.* = undefined;
    }

    /// Compile GLSL source to SPIR-V
    pub fn compile(self: Self, options: CompileOptions) CompileError!CompileResult {
        return switch (self.backend) {
            .glslc => self.compileWithGlslc(options),
            .glslang_validator => self.compileWithGlslang(options),
        };
    }

    fn compileWithGlslc(self: Self, options: CompileOptions) CompileError!CompileResult {
        // Build argument list
        var args = std.ArrayListUnmanaged([]const u8){};
        defer args.deinit(self.allocator);

        args.append(self.allocator, self.compiler_path) catch return CompileError.OutOfMemory;

        // Shader stage
        args.append(self.allocator, options.stage.toGlslcFlag()) catch return CompileError.OutOfMemory;

        // Target environment
        var target_buf: [32]u8 = undefined;
        const target_env = std.fmt.bufPrint(&target_buf, "--target-env=vulkan{s}", .{options.target_vulkan}) catch return CompileError.OutOfMemory;
        args.append(self.allocator, target_env) catch return CompileError.OutOfMemory;

        // Optimization
        if (options.optimization.toGlslcFlag()) |opt_flag| {
            args.append(self.allocator, opt_flag) catch return CompileError.OutOfMemory;
        }

        // Debug info
        if (options.debug_info) {
            args.append(self.allocator, "-g") catch return CompileError.OutOfMemory;
        }

        // Entry point
        var entry_buf: [128]u8 = undefined;
        const entry_arg = std.fmt.bufPrint(&entry_buf, "-fentry-point={s}", .{options.entry_point}) catch return CompileError.OutOfMemory;
        args.append(self.allocator, entry_arg) catch return CompileError.OutOfMemory;

        // Defines
        for (options.defines) |def| {
            var def_buf: [256]u8 = undefined;
            const def_arg = if (def.value) |val|
                std.fmt.bufPrint(&def_buf, "-D{s}={s}", .{ def.name, val }) catch return CompileError.OutOfMemory
            else
                std.fmt.bufPrint(&def_buf, "-D{s}", .{def.name}) catch return CompileError.OutOfMemory;
            args.append(self.allocator, def_arg) catch return CompileError.OutOfMemory;
        }

        // Include paths
        for (options.include_paths) |inc_path| {
            args.append(self.allocator, "-I") catch return CompileError.OutOfMemory;
            args.append(self.allocator, inc_path) catch return CompileError.OutOfMemory;
        }

        // Output to stdout
        args.append(self.allocator, "-o") catch return CompileError.OutOfMemory;
        args.append(self.allocator, "-") catch return CompileError.OutOfMemory;

        // Input from stdin
        args.append(self.allocator, "-") catch return CompileError.OutOfMemory;

        // Run compiler
        return self.runCompiler(args.items, options.source);
    }

    fn compileWithGlslang(self: Self, options: CompileOptions) CompileError!CompileResult {
        // Build argument list for glslangValidator
        var args = std.ArrayListUnmanaged([]const u8){};
        defer args.deinit(self.allocator);

        args.append(self.allocator, self.compiler_path) catch return CompileError.OutOfMemory;

        // SPIR-V output
        args.append(self.allocator, "-V") catch return CompileError.OutOfMemory;

        // Shader stage
        args.append(self.allocator, "-S") catch return CompileError.OutOfMemory;
        args.append(self.allocator, switch (options.stage) {
            .vertex => "vert",
            .fragment => "frag",
            .compute => "comp",
            .geometry => "geom",
            .tessellation_control => "tesc",
            .tessellation_evaluation => "tese",
            else => "comp",
        }) catch return CompileError.OutOfMemory;

        // Target environment
        var target_buf: [32]u8 = undefined;
        const target_env = std.fmt.bufPrint(&target_buf, "--target-env vulkan{s}", .{options.target_vulkan}) catch return CompileError.OutOfMemory;
        args.append(self.allocator, target_env) catch return CompileError.OutOfMemory;

        // Entry point
        args.append(self.allocator, "-e") catch return CompileError.OutOfMemory;
        args.append(self.allocator, options.entry_point) catch return CompileError.OutOfMemory;

        // Output to stdout
        args.append(self.allocator, "-o") catch return CompileError.OutOfMemory;
        args.append(self.allocator, "-") catch return CompileError.OutOfMemory;

        // Input from stdin
        args.append(self.allocator, "--stdin") catch return CompileError.OutOfMemory;

        return self.runCompiler(args.items, options.source);
    }

    fn runCompiler(self: Self, args: []const []const u8, source: []const u8) CompileError!CompileResult {
        var child = std.process.Child.init(args, self.allocator);
        child.stdin_behavior = .Pipe;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;

        child.spawn() catch return CompileError.ProcessError;

        // Write source to stdin
        if (child.stdin) |stdin| {
            stdin.writeAll(source) catch return CompileError.IoError;
            stdin.close();
            child.stdin = null;
        }

        // Read stdout (SPIR-V output)
        const stdout = child.stdout orelse return CompileError.ProcessError;
        const spirv_bytes = stdout.readToEndAlloc(self.allocator, 64 * 1024 * 1024) catch return CompileError.IoError;

        // Wait for completion
        const result = child.wait() catch return CompileError.ProcessError;

        // Check exit code
        const exit_code: u8 = switch (result) {
            .Exited => |code| code,
            else => 1, // Non-zero for non-exited termination
        };
        if (exit_code != 0) {
            self.allocator.free(spirv_bytes);
            return CompileError.CompilationFailed;
        }

        // Validate SPIR-V size
        if (spirv_bytes.len < 20 or spirv_bytes.len % 4 != 0) {
            self.allocator.free(spirv_bytes);
            return CompileError.CompilationFailed;
        }

        // Convert to u32 slice
        const spirv: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, spirv_bytes));

        return CompileResult{
            .spirv = spirv,
            .allocator = self.allocator,
        };
    }

    fn findCompiler(allocator: std.mem.Allocator, backend: CompilerBackend) ?[]const u8 {
        const exe_name = backend.executable();

        // Check if compiler is in PATH (platform-specific)
        const path_env = if (builtin.os.tag == .windows)
            std.process.getEnvVarOwned(allocator, "PATH") catch return null
        else
            std.posix.getenv("PATH") orelse return null;
        defer if (builtin.os.tag == .windows) allocator.free(path_env);

        const path_sep: u8 = if (builtin.os.tag == .windows) ';' else ':';
        var path_iter = std.mem.splitScalar(u8, path_env, path_sep);

        while (path_iter.next()) |dir| {
            // Check if executable exists in this directory
            var path_buf: [std.fs.max_path_bytes]u8 = undefined;
            const full_path = std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, exe_name }) catch continue;

            const file = std.fs.cwd().openFile(full_path, .{}) catch continue;
            file.close();

            // Return static path based on backend
            return exe_name;
        }

        // Check common installation paths
        const common_paths = switch (builtin.os.tag) {
            .linux => &[_][]const u8{
                "/usr/bin/",
                "/usr/local/bin/",
                "/opt/vulkan/bin/",
            },
            .macos => &[_][]const u8{
                "/usr/local/bin/",
                "/opt/homebrew/bin/",
            },
            .windows => &[_][]const u8{
                "C:\\VulkanSDK\\Bin\\",
            },
            else => &[_][]const u8{},
        };

        for (common_paths) |base| {
            var path_buf: [std.fs.max_path_bytes]u8 = undefined;
            const full_path = std.fmt.bufPrint(&path_buf, "{s}{s}", .{ base, exe_name }) catch continue;

            const file = std.fs.cwd().openFile(full_path, .{}) catch continue;
            file.close();
            return exe_name;
        }

        return null;
    }
};

/// Check if a GLSL compiler is available on the system
pub fn isCompilerAvailable() bool {
    const allocator = std.heap.page_allocator;
    var compiler = Compiler.init(allocator) catch return false;
    compiler.deinit();
    return true;
}

/// Convenience function to compile GLSL to SPIR-V
pub fn compile(allocator: std.mem.Allocator, options: CompileOptions) CompileError!CompileResult {
    var compiler = try Compiler.init(allocator);
    defer compiler.deinit();
    return compiler.compile(options);
}

/// Compile GLSL file to SPIR-V
pub fn compileFile(allocator: std.mem.Allocator, path: []const u8, stage: ShaderStage) CompileError!CompileResult {
    const file = std.fs.cwd().openFile(path, .{}) catch return CompileError.IoError;
    defer file.close();

    const source = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return CompileError.IoError;
    defer allocator.free(source);

    return compile(allocator, .{
        .source = source,
        .stage = stage,
    });
}

// =============================================================================
// HLSL Compilation Support
// =============================================================================

/// HLSL shader model
pub const HlslShaderModel = enum {
    sm_6_0,
    sm_6_1,
    sm_6_2,
    sm_6_3,
    sm_6_4,
    sm_6_5,
    sm_6_6,
    sm_6_7,

    pub fn toString(self: HlslShaderModel) []const u8 {
        return switch (self) {
            .sm_6_0 => "6_0",
            .sm_6_1 => "6_1",
            .sm_6_2 => "6_2",
            .sm_6_3 => "6_3",
            .sm_6_4 => "6_4",
            .sm_6_5 => "6_5",
            .sm_6_6 => "6_6",
            .sm_6_7 => "6_7",
        };
    }
};

/// HLSL target profile for DXC
pub const HlslProfile = struct {
    stage: ShaderStage,
    shader_model: HlslShaderModel,

    pub fn toProfileString(self: HlslProfile, buf: []u8) []const u8 {
        const stage_prefix = switch (self.stage) {
            .compute => "cs",
            .vertex => "vs",
            .fragment => "ps",
            .geometry => "gs",
            .tessellation_control => "hs",
            .tessellation_evaluation => "ds",
            .ray_generation => "lib",
            .ray_intersection => "lib",
            .ray_any_hit => "lib",
            .ray_closest_hit => "lib",
            .ray_miss => "lib",
            .task => "as",
            .mesh => "ms",
        };
        return std.fmt.bufPrint(buf, "{s}_{s}", .{ stage_prefix, self.shader_model.toString() }) catch "cs_6_0";
    }
};

/// HLSL compilation options
pub const HlslCompileOptions = struct {
    /// HLSL source code
    source: []const u8,
    /// Shader stage
    stage: ShaderStage,
    /// Entry point name
    entry_point: []const u8 = "main",
    /// Shader model
    shader_model: HlslShaderModel = .sm_6_0,
    /// Optimization level
    optimization: OptimizationLevel = .performance,
    /// Generate debug info
    debug_info: bool = false,
    /// Target Vulkan version
    target_vulkan: []const u8 = "1.0",
    /// Additional defines
    defines: []const Define = &.{},
    /// Include paths
    include_paths: []const []const u8 = &.{},
    /// Enable 16-bit types
    enable_16bit_types: bool = false,
    /// Matrix row-major layout
    row_major_matrices: bool = false,
};

/// HLSL compiler (DXC) wrapper
pub const HlslCompiler = struct {
    allocator: std.mem.Allocator,
    dxc_path: []const u8,

    const Self = @This();

    /// Find and initialize DXC compiler
    pub fn init(allocator: std.mem.Allocator) CompileError!Self {
        if (findDxc(allocator)) |path| {
            return Self{
                .allocator = allocator,
                .dxc_path = path,
            };
        }
        return CompileError.CompilerNotFound;
    }

    /// Initialize with specific DXC path
    pub fn initWithPath(allocator: std.mem.Allocator, path: []const u8) Self {
        return Self{
            .allocator = allocator,
            .dxc_path = path,
        };
    }

    pub fn deinit(self: *Self) void {
        self.* = undefined;
    }

    /// Compile HLSL to SPIR-V
    pub fn compile(self: Self, options: HlslCompileOptions) CompileError!CompileResult {
        var args = std.ArrayListUnmanaged([]const u8){};
        defer args.deinit();

        args.append(self.dxc_path) catch return CompileError.OutOfMemory;

        // SPIR-V output
        args.append("-spirv") catch return CompileError.OutOfMemory;

        // Target environment
        var target_buf: [32]u8 = undefined;
        const target_env = std.fmt.bufPrint(&target_buf, "-fspv-target-env=vulkan{s}", .{options.target_vulkan}) catch return CompileError.OutOfMemory;
        args.append(target_env) catch return CompileError.OutOfMemory;

        // Shader profile
        var profile_buf: [32]u8 = undefined;
        const profile = HlslProfile{ .stage = options.stage, .shader_model = options.shader_model };
        const profile_str = profile.toProfileString(&profile_buf);
        args.append("-T") catch return CompileError.OutOfMemory;
        args.append(profile_str) catch return CompileError.OutOfMemory;

        // Entry point
        args.append("-E") catch return CompileError.OutOfMemory;
        args.append(options.entry_point) catch return CompileError.OutOfMemory;

        // Optimization
        switch (options.optimization) {
            .none => args.append("-Od") catch return CompileError.OutOfMemory,
            .size => args.append("-Os") catch return CompileError.OutOfMemory,
            .performance => args.append("-O3") catch return CompileError.OutOfMemory,
        }

        // Debug info
        if (options.debug_info) {
            args.append("-Zi") catch return CompileError.OutOfMemory;
        }

        // 16-bit types
        if (options.enable_16bit_types) {
            args.append("-enable-16bit-types") catch return CompileError.OutOfMemory;
        }

        // Matrix layout
        if (options.row_major_matrices) {
            args.append("-Zpr") catch return CompileError.OutOfMemory;
        }

        // Defines
        for (options.defines) |def| {
            var def_buf: [256]u8 = undefined;
            const def_arg = if (def.value) |val|
                std.fmt.bufPrint(&def_buf, "-D{s}={s}", .{ def.name, val }) catch return CompileError.OutOfMemory
            else
                std.fmt.bufPrint(&def_buf, "-D{s}", .{def.name}) catch return CompileError.OutOfMemory;
            args.append(def_arg) catch return CompileError.OutOfMemory;
        }

        // Include paths
        for (options.include_paths) |inc_path| {
            args.append("-I") catch return CompileError.OutOfMemory;
            args.append(inc_path) catch return CompileError.OutOfMemory;
        }

        // Output to stdout
        args.append("-Fo") catch return CompileError.OutOfMemory;
        args.append("-") catch return CompileError.OutOfMemory;

        return self.runDxc(args.items, options.source);
    }

    fn runDxc(self: Self, args: []const []const u8, source: []const u8) CompileError!CompileResult {
        var child = std.process.Child.init(args, self.allocator);
        child.stdin_behavior = .Pipe;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;

        child.spawn() catch return CompileError.ProcessError;

        // Write source to stdin
        if (child.stdin) |stdin| {
            stdin.writeAll(source) catch return CompileError.IoError;
            stdin.close();
            child.stdin = null;
        }

        // Read stdout (SPIR-V output)
        const stdout = child.stdout orelse return CompileError.ProcessError;
        const spirv_bytes = stdout.readToEndAlloc(self.allocator, 64 * 1024 * 1024) catch return CompileError.IoError;

        // Wait for completion
        const result = child.wait() catch return CompileError.ProcessError;

        // Check exit code
        const exit_code: u8 = switch (result) {
            .Exited => |code| code,
            else => 1, // Non-zero for non-exited termination
        };
        if (exit_code != 0) {
            self.allocator.free(spirv_bytes);
            return CompileError.CompilationFailed;
        }

        // Validate SPIR-V size
        if (spirv_bytes.len < 20 or spirv_bytes.len % 4 != 0) {
            self.allocator.free(spirv_bytes);
            return CompileError.CompilationFailed;
        }

        // Convert to u32 slice
        const spirv: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, spirv_bytes));

        return CompileResult{
            .spirv = spirv,
            .allocator = self.allocator,
        };
    }

    fn findDxc(allocator: std.mem.Allocator) ?[]const u8 {
        const exe_name = "dxc";

        // Check if DXC is in PATH (platform-specific)
        const path_env = if (builtin.os.tag == .windows)
            std.process.getEnvVarOwned(allocator, "PATH") catch return null
        else
            std.posix.getenv("PATH") orelse return null;
        defer if (builtin.os.tag == .windows) allocator.free(path_env);

        const path_sep: u8 = if (builtin.os.tag == .windows) ';' else ':';
        var path_iter = std.mem.splitScalar(u8, path_env, path_sep);

        while (path_iter.next()) |dir| {
            var path_buf: [std.fs.max_path_bytes]u8 = undefined;
            const full_path = std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, exe_name }) catch continue;

            const file = std.fs.cwd().openFile(full_path, .{}) catch continue;
            file.close();
            return exe_name;
        }

        // Check common installation paths
        const common_paths = switch (builtin.os.tag) {
            .linux => &[_][]const u8{
                "/usr/bin/",
                "/usr/local/bin/",
            },
            .macos => &[_][]const u8{
                "/usr/local/bin/",
                "/opt/homebrew/bin/",
            },
            .windows => &[_][]const u8{
                "C:\\VulkanSDK\\Bin\\",
            },
            else => &[_][]const u8{},
        };

        for (common_paths) |base| {
            var path_buf: [std.fs.max_path_bytes]u8 = undefined;
            const full_path = std.fmt.bufPrint(&path_buf, "{s}{s}", .{ base, exe_name }) catch continue;

            const file = std.fs.cwd().openFile(full_path, .{}) catch continue;
            file.close();
            return exe_name;
        }

        return null;
    }
};

/// Check if HLSL compiler (DXC) is available
pub fn isHlslCompilerAvailable() bool {
    const allocator = std.heap.page_allocator;
    var compiler = HlslCompiler.init(allocator) catch return false;
    compiler.deinit();
    return true;
}

// =============================================================================
// Compiler Availability Detection
// =============================================================================

/// Compiler availability information
pub const CompilerInfo = struct {
    /// Whether GLSL compiler is available
    glsl_available: bool = false,
    /// Whether HLSL compiler (DXC) is available
    hlsl_available: bool = false,
    /// GLSL compiler backend in use
    glsl_backend: ?CompilerBackend = null,
    /// GLSL compiler path
    glsl_path: ?[]const u8 = null,
    /// DXC compiler path
    dxc_path: ?[]const u8 = null,
    /// DXC version (if detected)
    dxc_version: ?[]const u8 = null,
    /// Maximum supported shader model
    max_shader_model: ?HlslShaderModel = null,

    pub fn format(
        self: CompilerInfo,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.writeAll("CompilerInfo{\n");
        try writer.print("  GLSL: {s}\n", .{if (self.glsl_available) "available" else "not found"});
        if (self.glsl_backend) |backend| {
            try writer.print("    Backend: {s}\n", .{backend.executable()});
        }
        try writer.print("  HLSL (DXC): {s}\n", .{if (self.hlsl_available) "available" else "not found"});
        if (self.dxc_version) |ver| {
            try writer.print("    Version: {s}\n", .{ver});
        }
        if (self.max_shader_model) |sm| {
            try writer.print("    Max SM: {s}\n", .{sm.toString()});
        }
        try writer.writeAll("}");
    }
};

/// Detect all available shader compilers
pub fn detectCompilers(allocator: std.mem.Allocator) CompilerInfo {
    var info = CompilerInfo{};

    // Check GLSL compiler
    if (Compiler.init(allocator)) |compiler| {
        info.glsl_available = true;
        info.glsl_backend = compiler.backend;
        info.glsl_path = compiler.compiler_path;
    } else |_| {}

    // Check HLSL compiler
    if (HlslCompiler.init(allocator)) |compiler| {
        info.hlsl_available = true;
        info.dxc_path = compiler.dxc_path;

        // Try to detect DXC version
        info.dxc_version = detectDxcVersion(allocator, compiler.dxc_path);
        info.max_shader_model = detectMaxShaderModel(allocator, compiler.dxc_path);
    } else |_| {}

    return info;
}

fn detectDxcVersion(allocator: std.mem.Allocator, dxc_path: []const u8) ?[]const u8 {
    const args = [_][]const u8{ dxc_path, "--version" };
    var child = std.process.Child.init(&args, allocator);
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    child.spawn() catch return null;

    const stdout = child.stdout orelse return null;
    const output = stdout.readToEndAlloc(allocator, 4096) catch return null;

    _ = child.wait() catch return null;

    // Parse version from output (typically "dxcompiler.dll: X.X.X.X" or similar)
    if (std.mem.indexOf(u8, output, "version")) |idx| {
        const start = idx + 8;
        if (start < output.len) {
            var end = start;
            while (end < output.len and (output[end] != '\n' and output[end] != '\r')) : (end += 1) {}
            if (end > start) {
                return output[start..end];
            }
        }
    }

    allocator.free(output);
    return null;
}

fn detectMaxShaderModel(allocator: std.mem.Allocator, dxc_path: []const u8) ?HlslShaderModel {
    // Try compiling with highest shader models to detect support
    const test_shader = "float4 main() : SV_Target { return 0; }";
    const shader_models = [_]HlslShaderModel{
        .sm_6_7, .sm_6_6, .sm_6_5, .sm_6_4, .sm_6_3, .sm_6_2, .sm_6_1, .sm_6_0,
    };

    for (shader_models) |sm| {
        var profile_buf: [32]u8 = undefined;
        const profile = HlslProfile{ .stage = .fragment, .shader_model = sm };
        const profile_str = profile.toProfileString(&profile_buf);

        const args = [_][]const u8{
            dxc_path,
            "-T",
            profile_str,
            "-E",
            "main",
            "-Fo",
            "-",
            "-",
        };

        var child = std.process.Child.init(&args, allocator);
        child.stdin_behavior = .Pipe;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;

        child.spawn() catch continue;

        if (child.stdin) |stdin| {
            stdin.writeAll(test_shader) catch |e| {
                std.log.debug("glslangValidator probe: failed to write test shader: {s}", .{@errorName(e)});
            };
            stdin.close();
            child.stdin = null;
        }

        const result = child.wait() catch continue;
        if (result.Exited == 0) {
            return sm;
        }
    }

    return null;
}

/// Compile HLSL to SPIR-V
pub fn compileHlsl(allocator: std.mem.Allocator, options: HlslCompileOptions) CompileError!CompileResult {
    var compiler = try HlslCompiler.init(allocator);
    defer compiler.deinit();
    return compiler.compile(options);
}

// =============================================================================
// Unified Shader Compiler
// =============================================================================

/// Source language for shader compilation
pub const SourceLanguage = enum {
    glsl,
    hlsl,
};

/// Unified shader compilation options
pub const UnifiedCompileOptions = struct {
    /// Shader source code
    source: []const u8,
    /// Source language
    language: SourceLanguage,
    /// Shader stage
    stage: ShaderStage,
    /// Entry point name
    entry_point: []const u8 = "main",
    /// Optimization level
    optimization: OptimizationLevel = .performance,
    /// Generate debug info
    debug_info: bool = false,
    /// Target Vulkan version
    target_vulkan: []const u8 = "1.0",
    /// Additional defines
    defines: []const Define = &.{},
    /// Include paths
    include_paths: []const []const u8 = &.{},
};

/// Unified shader compiler supporting both GLSL and HLSL
pub const UnifiedCompiler = struct {
    allocator: std.mem.Allocator,
    glsl_compiler: ?Compiler,
    hlsl_compiler: ?HlslCompiler,

    const Self = @This();

    /// Initialize with all available compilers
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .glsl_compiler = Compiler.init(allocator) catch null,
            .hlsl_compiler = HlslCompiler.init(allocator) catch null,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.glsl_compiler) |*c| c.deinit();
        if (self.hlsl_compiler) |*c| c.deinit();
    }

    /// Check if a language is supported
    pub fn supportsLanguage(self: Self, lang: SourceLanguage) bool {
        return switch (lang) {
            .glsl => self.glsl_compiler != null,
            .hlsl => self.hlsl_compiler != null,
        };
    }

    /// Compile shader to SPIR-V
    pub fn compile(self: Self, options: UnifiedCompileOptions) CompileError!CompileResult {
        switch (options.language) {
            .glsl => {
                const compiler = self.glsl_compiler orelse return CompileError.CompilerNotFound;
                return compiler.compile(.{
                    .source = options.source,
                    .stage = options.stage,
                    .entry_point = options.entry_point,
                    .optimization = options.optimization,
                    .debug_info = options.debug_info,
                    .target_vulkan = options.target_vulkan,
                    .defines = options.defines,
                    .include_paths = options.include_paths,
                });
            },
            .hlsl => {
                const compiler = self.hlsl_compiler orelse return CompileError.CompilerNotFound;
                return compiler.compile(.{
                    .source = options.source,
                    .stage = options.stage,
                    .entry_point = options.entry_point,
                    .optimization = options.optimization,
                    .debug_info = options.debug_info,
                    .target_vulkan = options.target_vulkan,
                    .defines = options.defines,
                    .include_paths = options.include_paths,
                });
            },
        }
    }

    /// Get available languages
    pub fn getAvailableLanguages(self: Self) []const SourceLanguage {
        var langs: [2]SourceLanguage = undefined;
        var count: usize = 0;

        if (self.glsl_compiler != null) {
            langs[count] = .glsl;
            count += 1;
        }
        if (self.hlsl_compiler != null) {
            langs[count] = .hlsl;
            count += 1;
        }

        return langs[0..count];
    }
};

/// Convenience function to compile any shader
pub fn compileShader(allocator: std.mem.Allocator, options: UnifiedCompileOptions) CompileError!CompileResult {
    var compiler = UnifiedCompiler.init(allocator);
    defer compiler.deinit();
    return compiler.compile(options);
}

/// Detect source language from file extension
pub fn detectLanguage(path: []const u8) ?SourceLanguage {
    const ext = std.fs.path.extension(path);

    // GLSL extensions
    if (std.mem.eql(u8, ext, ".glsl") or
        std.mem.eql(u8, ext, ".vert") or
        std.mem.eql(u8, ext, ".frag") or
        std.mem.eql(u8, ext, ".comp") or
        std.mem.eql(u8, ext, ".geom") or
        std.mem.eql(u8, ext, ".tesc") or
        std.mem.eql(u8, ext, ".tese"))
    {
        return .glsl;
    }

    // HLSL extensions
    if (std.mem.eql(u8, ext, ".hlsl") or
        std.mem.eql(u8, ext, ".fx") or
        std.mem.eql(u8, ext, ".csh") or
        std.mem.eql(u8, ext, ".vsh") or
        std.mem.eql(u8, ext, ".psh"))
    {
        return .hlsl;
    }

    return null;
}

// =============================================================================
// Shader Compilation Cache
// =============================================================================

/// Shader cache configuration
pub const ShaderCacheConfig = struct {
    /// Cache directory path
    cache_dir: []const u8 = ".shader_cache",
    /// Maximum cache size in bytes (0 = unlimited)
    max_cache_size: u64 = 256 * 1024 * 1024, // 256 MB default
    /// Cache entry TTL in seconds (0 = never expire)
    ttl_seconds: u64 = 0,
    /// Whether to validate cached SPIR-V before returning
    validate_on_load: bool = true,
    /// Whether to create cache directory if missing
    create_dir: bool = true,
};

/// Shader compilation cache for GLSL/HLSL to SPIR-V
///
/// Caches compiled SPIR-V bytecode on disk to avoid repeated compilations.
/// Uses content hashing to identify shader sources uniquely.
///
/// ## Example
/// ```zig
/// var cache = try ShaderCache.init(allocator, .{});
/// defer cache.deinit();
///
/// // Compile with caching
/// const spirv = try cache.compileGlsl(.{
///     .source = @embedFile("shader.glsl"),
///     .stage = .compute,
/// });
/// defer allocator.free(spirv);
/// ```
pub const ShaderCache = struct {
    allocator: std.mem.Allocator,
    config: ShaderCacheConfig,
    glsl_compiler: ?Compiler = null,
    hlsl_compiler: ?HlslCompiler = null,

    /// Cache statistics
    stats: CacheStats = .{},

    const Self = @This();

    /// Cache statistics
    pub const CacheStats = struct {
        /// Number of cache hits
        hits: u64 = 0,
        /// Number of cache misses
        misses: u64 = 0,
        /// Number of compilations performed
        compilations: u64 = 0,
        /// Total bytes read from cache
        bytes_read: u64 = 0,
        /// Total bytes written to cache
        bytes_written: u64 = 0,
        /// Number of cache validation failures
        validation_failures: u64 = 0,

        pub fn hitRate(self: CacheStats) f64 {
            const total = self.hits + self.misses;
            if (total == 0) return 1.0;
            return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
        }
    };

    /// Initialize shader cache
    pub fn init(allocator: std.mem.Allocator, config: ShaderCacheConfig) !Self {
        var self = Self{
            .allocator = allocator,
            .config = config,
        };

        // Create cache directory if needed
        if (config.create_dir) {
            std.fs.cwd().makePath(config.cache_dir) catch |e| {
                if (e != error.PathAlreadyExists) {
                    // Non-fatal - continue without caching
                    std.log.warn("[ShaderCache] Failed to create cache directory: {}", .{e});
                }
            };
        }

        // Try to initialize compilers
        self.glsl_compiler = Compiler.init(allocator) catch null;
        self.hlsl_compiler = HlslCompiler.init(allocator) catch null;

        return self;
    }

    /// Deinitialize and clean up
    pub fn deinit(self: *Self) void {
        if (self.glsl_compiler) |*c| c.deinit();
        if (self.hlsl_compiler) |*c| c.deinit();
        self.* = undefined;
    }

    /// Compile GLSL to SPIR-V with caching
    pub fn compileGlsl(self: *Self, options: CompileOptions) CompileError!CompileResult {
        const compiler = self.glsl_compiler orelse return CompileError.CompilerNotFound;

        // Generate cache key
        const cache_key = self.generateCacheKey(.glsl, options.source, options.stage, options.entry_point);

        // Try to load from cache
        if (self.loadFromCache(cache_key)) |cached| {
            self.stats.hits += 1;
            self.stats.bytes_read += cached.spirv.len * @sizeOf(u32);
            return cached;
        }
        self.stats.misses += 1;

        // Compile
        const result = try compiler.compile(options);
        self.stats.compilations += 1;

        // Store in cache
        self.saveToCache(cache_key, result);
        self.stats.bytes_written += result.spirv.len * @sizeOf(u32);

        return result;
    }

    /// Compile HLSL to SPIR-V with caching
    pub fn compileHlsl(self: *Self, options: HlslCompileOptions) CompileError!CompileResult {
        const compiler = self.hlsl_compiler orelse return CompileError.CompilerNotFound;

        // Generate cache key
        const cache_key = self.generateCacheKey(.hlsl, options.source, options.stage, options.entry_point);

        // Try to load from cache
        if (self.loadFromCache(cache_key)) |cached| {
            self.stats.hits += 1;
            self.stats.bytes_read += cached.spirv.len * @sizeOf(u32);
            return cached;
        }
        self.stats.misses += 1;

        // Compile
        const result = try compiler.compile(options);
        self.stats.compilations += 1;

        // Store in cache
        self.saveToCache(cache_key, result);
        self.stats.bytes_written += result.spirv.len * @sizeOf(u32);

        return result;
    }

    /// Generate a cache key from shader content
    fn generateCacheKey(
        self: *Self,
        language: SourceLanguage,
        source: []const u8,
        stage: ShaderStage,
        entry_point: []const u8,
    ) [32]u8 {
        _ = self;
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(source);
        hasher.update(entry_point);
        hasher.update(@tagName(language));
        hasher.update(@tagName(stage));

        const hash = hasher.final();

        // Convert to hex string for filename
        var key: [32]u8 = undefined;
        _ = std.fmt.bufPrint(&key, "{x:0>16}{s}", .{ hash, stage.fileExtension() }) catch |e| {
            std.log.warn("ShaderCache.generateKey: failed to format cache key: {s}", .{@errorName(e)});
            @memset(&key, '0'); // Return zeroed key as fallback
        };
        return key;
    }

    /// Load SPIR-V from cache
    fn loadFromCache(self: *Self, key: [32]u8) ?CompileResult {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&path_buf, "{s}/{s}.spv", .{ self.config.cache_dir, key[0..16] }) catch return null;

        const file = std.fs.cwd().openFile(path, .{}) catch return null;
        defer file.close();

        // Check TTL if configured
        if (self.config.ttl_seconds > 0) {
            const stat = file.stat() catch return null;
            const now = std.time.timestamp();
            const file_age: u64 = @intCast(now - @as(i64, @intCast(stat.mtime)));
            if (file_age > self.config.ttl_seconds * std.time.ns_per_s) {
                return null; // Cache entry expired
            }
        }

        // Read file size
        const stat = file.stat() catch return null;
        if (stat.size < 20 or stat.size % 4 != 0) return null; // Invalid SPIR-V

        const word_count = stat.size / 4;
        const spirv = self.allocator.alloc(u32, word_count) catch return null;
        errdefer self.allocator.free(spirv);

        const bytes_read = file.readAll(std.mem.sliceAsBytes(spirv)) catch {
            self.allocator.free(spirv);
            return null;
        };
        if (bytes_read != stat.size) {
            self.allocator.free(spirv);
            return null;
        }

        // Validate SPIR-V if configured
        if (self.config.validate_on_load) {
            if (!isValidSpirv(spirv)) {
                self.allocator.free(spirv);
                self.stats.validation_failures += 1;
                return null;
            }
        }

        return CompileResult{
            .spirv = spirv,
            .allocator = self.allocator,
        };
    }

    /// Save SPIR-V to cache
    fn saveToCache(self: *Self, key: [32]u8, result: CompileResult) void {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&path_buf, "{s}/{s}.spv", .{ self.config.cache_dir, key[0..16] }) catch return;

        const file = std.fs.cwd().createFile(path, .{}) catch return;
        defer file.close();

        file.writeAll(std.mem.sliceAsBytes(result.spirv)) catch return;
    }

    /// Clear all cached shaders
    pub fn clearCache(self: *Self) void {
        var dir = std.fs.cwd().openDir(self.config.cache_dir, .{ .iterate = true }) catch return;
        defer dir.close();

        var iter = dir.iterate();
        while (iter.next() catch null) |entry| {
            if (std.mem.endsWith(u8, entry.name, ".spv")) {
                dir.deleteFile(entry.name) catch continue;
            }
        }
    }

    /// Get cache statistics
    pub fn getStats(self: Self) CacheStats {
        return self.stats;
    }

    /// Reset cache statistics
    pub fn resetStats(self: *Self) void {
        self.stats = .{};
    }
};

/// Convenience function to compile with caching (uses global cache)
var global_cache: ?ShaderCache = null;

pub fn compileWithCache(allocator: std.mem.Allocator, options: CompileOptions) CompileError!CompileResult {
    if (global_cache == null) {
        global_cache = ShaderCache.init(allocator, .{}) catch null;
    }
    if (global_cache) |*cache| {
        return cache.compileGlsl(options);
    }
    // Fall back to non-cached compilation
    return compile(allocator, options);
}

// =============================================================================
// Tests
// =============================================================================

test "HlslShaderModel toString" {
    try std.testing.expectEqualStrings("6_0", HlslShaderModel.sm_6_0.toString());
    try std.testing.expectEqualStrings("6_5", HlslShaderModel.sm_6_5.toString());
}

test "HlslProfile toProfileString" {
    var buf: [32]u8 = undefined;
    const profile = HlslProfile{ .stage = .compute, .shader_model = .sm_6_0 };
    try std.testing.expectEqualStrings("cs_6_0", profile.toProfileString(&buf));
}

test "detectLanguage" {
    try std.testing.expectEqual(SourceLanguage.glsl, detectLanguage("shader.glsl").?);
    try std.testing.expectEqual(SourceLanguage.glsl, detectLanguage("shader.comp").?);
    try std.testing.expectEqual(SourceLanguage.hlsl, detectLanguage("shader.hlsl").?);
    try std.testing.expect(detectLanguage("shader.unknown") == null);
}

test "ShaderStage file extensions" {
    try std.testing.expectEqualStrings(".comp", ShaderStage.compute.fileExtension());
    try std.testing.expectEqualStrings(".vert", ShaderStage.vertex.fileExtension());
    try std.testing.expectEqualStrings(".frag", ShaderStage.fragment.fileExtension());
}

test "ShaderStage glslc flags" {
    try std.testing.expectEqualStrings("-fshader-stage=compute", ShaderStage.compute.toGlslcFlag());
    try std.testing.expectEqualStrings("-fshader-stage=vertex", ShaderStage.vertex.toGlslcFlag());
}

test "OptimizationLevel flags" {
    try std.testing.expectEqualStrings("-O0", OptimizationLevel.none.toGlslcFlag().?);
    try std.testing.expectEqualStrings("-O", OptimizationLevel.performance.toGlslcFlag().?);
}

test "GlslVersion toString" {
    try std.testing.expectEqualStrings("450", GlslVersion.@"450".toString());
    try std.testing.expectEqualStrings("460", GlslVersion.@"460".toString());
}

test "CompileOptions defaults" {
    const opts = CompileOptions{
        .source = "void main() {}",
        .stage = .compute,
    };
    try std.testing.expectEqualStrings("main", opts.entry_point);
    try std.testing.expectEqual(GlslVersion.@"450", opts.version);
    try std.testing.expectEqual(OptimizationLevel.performance, opts.optimization);
}
