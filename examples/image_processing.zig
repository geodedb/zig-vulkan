//! Image Processing Example
//!
//! Demonstrates GPU image processing using Vulkan compute shaders.
//! Includes common operations: blur, edge detection, color transformation.
//!
//! This example shows the buffer setup and dispatch configuration
//! for typical image processing workloads.

const std = @import("std");
const vulkan = @import("vulkan");

/// Image dimensions
const IMAGE_WIDTH: u32 = 1920;
const IMAGE_HEIGHT: u32 = 1080;
const CHANNELS: u32 = 4; // RGBA

/// Workgroup size for image processing
const WORKGROUP_SIZE_X: u32 = 16;
const WORKGROUP_SIZE_Y: u32 = 16;

/// Push constants for image processing
const ImagePushConstants = extern struct {
    width: u32,
    height: u32,
    filter_radius: u32,
    strength: f32,
};

/// Image filter types
const FilterType = enum {
    blur,
    sharpen,
    edge_detect,
    grayscale,
    brightness,
    saturation,

    fn getName(self: FilterType) []const u8 {
        return switch (self) {
            .blur => "Gaussian Blur",
            .sharpen => "Sharpen",
            .edge_detect => "Edge Detection (Sobel)",
            .grayscale => "Grayscale",
            .brightness => "Brightness Adjust",
            .saturation => "Saturation Adjust",
        };
    }

    fn getWorkloadDescription(self: FilterType) []const u8 {
        return switch (self) {
            .blur => "Memory bandwidth limited, reads neighborhood",
            .sharpen => "Memory + compute, convolution kernel",
            .edge_detect => "Compute intensive, gradient calculation",
            .grayscale => "Simple per-pixel, memory limited",
            .brightness => "Simple per-pixel, memory limited",
            .saturation => "Per-pixel color space conversion",
        };
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Image Processing Example\n", .{});
    std.debug.print("========================\n\n", .{});

    // Check availability
    if (!vulkan.isAvailable()) {
        std.debug.print("Vulkan not available\n", .{});
        return;
    }

    // Create context
    var ctx = vulkan.Context.init(allocator, .{
        .app_name = "image_processing_example",
        .enable_validation = @import("builtin").mode == .Debug,
    }) catch |e| {
        std.debug.print("Failed to create context: {s}\n", .{vulkan.vk.describe(e)});
        return;
    };
    defer ctx.deinit();

    const device_info = ctx.deviceInfo();
    std.debug.print("Device: {s}\n", .{device_info.getName()});
    std.debug.print("Shared memory: {} KB\n\n", .{device_info.max_compute_shared_memory_size / 1024});

    // Image configuration
    const pixel_count = IMAGE_WIDTH * IMAGE_HEIGHT;
    const image_size = pixel_count * CHANNELS;
    const image_bytes = image_size * @sizeOf(u8);

    std.debug.print("Image Configuration:\n", .{});
    std.debug.print("  Resolution: {}x{}\n", .{ IMAGE_WIDTH, IMAGE_HEIGHT });
    std.debug.print("  Channels: {} (RGBA)\n", .{CHANNELS});
    std.debug.print("  Pixel count: {} ({} megapixels)\n", .{ pixel_count, pixel_count / 1_000_000 });
    std.debug.print("  Memory: {} bytes ({} MB)\n\n", .{ image_bytes, image_bytes / (1024 * 1024) });

    // Create buffers
    std.debug.print("Creating GPU buffers...\n", .{});

    var input_image = try vulkan.Buffer.init(&ctx, image_bytes, .staging);
    defer input_image.deinit(&ctx);

    var output_image = try vulkan.Buffer.init(&ctx, image_bytes, .readback);
    defer output_image.deinit(&ctx);

    // For filters that need temporary storage (e.g., separable blur)
    var temp_buffer = try vulkan.Buffer.init(&ctx, image_bytes, .storage);
    defer temp_buffer.deinit(&ctx);

    std.debug.print("  Input:  {} bytes (staging)\n", .{image_bytes});
    std.debug.print("  Output: {} bytes (readback)\n", .{image_bytes});
    std.debug.print("  Temp:   {} bytes (storage)\n\n", .{image_bytes});

    // Dispatch configuration
    const groups_x = (IMAGE_WIDTH + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
    const groups_y = (IMAGE_HEIGHT + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;

    std.debug.print("Dispatch Configuration:\n", .{});
    std.debug.print("  Workgroup size: {}x{}x1\n", .{ WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y });
    std.debug.print("  Dispatch groups: {}x{}x1\n", .{ groups_x, groups_y });
    std.debug.print("  Total threads: {}\n\n", .{groups_x * groups_y * WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y});

    // Print filter information
    std.debug.print("Available Image Filters:\n", .{});
    std.debug.print("-" ** 60 ++ "\n", .{});

    const filters = [_]FilterType{ .blur, .sharpen, .edge_detect, .grayscale, .brightness, .saturation };

    for (filters) |filter| {
        std.debug.print("\n{s}:\n", .{filter.getName()});
        std.debug.print("  {s}\n", .{filter.getWorkloadDescription()});

        // Estimate bandwidth/compute requirements
        const read_bytes = switch (filter) {
            .blur => image_bytes * 9, // 3x3 kernel reads 9 pixels
            .sharpen => image_bytes * 9,
            .edge_detect => image_bytes * 9, // Sobel requires 3x3
            .grayscale, .brightness, .saturation => image_bytes,
        };
        const write_bytes = image_bytes;

        std.debug.print("  Memory reads:  ~{} MB\n", .{read_bytes / (1024 * 1024)});
        std.debug.print("  Memory writes: ~{} MB\n", .{write_bytes / (1024 * 1024)});
    }

    // Example push constants
    const blur_params = ImagePushConstants{
        .width = IMAGE_WIDTH,
        .height = IMAGE_HEIGHT,
        .filter_radius = 3,
        .strength = 1.0,
    };

    std.debug.print("\n", .{});
    std.debug.print("Example GLSL Shader (Gaussian Blur):\n", .{});
    std.debug.print("-" ** 60 ++ "\n", .{});
    std.debug.print(
        \\#version 450
        \\layout(local_size_x = 16, local_size_y = 16) in;
        \\
        \\layout(set = 0, binding = 0) readonly buffer Input {{ vec4 pixels[]; }};
        \\layout(set = 0, binding = 1) writeonly buffer Output {{ vec4 result[]; }};
        \\
        \\layout(push_constant) uniform PC {{
        \\    uint width, height, radius;
        \\    float strength;
        \\}};
        \\
        \\void main() {{
        \\    uvec2 pos = gl_GlobalInvocationID.xy;
        \\    if (pos.x >= width || pos.y >= height) return;
        \\
        \\    vec4 sum = vec4(0);
        \\    float weight = 0;
        \\
        \\    for (int dy = -int(radius); dy <= int(radius); dy++) {{
        \\        for (int dx = -int(radius); dx <= int(radius); dx++) {{
        \\            ivec2 p = ivec2(pos) + ivec2(dx, dy);
        \\            if (p.x >= 0 && p.x < width && p.y >= 0 && p.y < height) {{
        \\                float w = exp(-(dx*dx + dy*dy) / (2.0 * radius * radius));
        \\                sum += pixels[p.y * width + p.x] * w;
        \\                weight += w;
        \\            }}
        \\        }}
        \\    }}
        \\
        \\    result[pos.y * width + pos.x] = sum / weight;
        \\}}
        \\
    , .{});

    std.debug.print("\nPush constants for blur: width={}, height={}, radius={}, strength={d:.1}\n", .{
        blur_params.width,
        blur_params.height,
        blur_params.filter_radius,
        blur_params.strength,
    });

    std.debug.print("\nExample completed.\n", .{});
}
