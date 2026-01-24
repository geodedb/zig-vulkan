#version 450

// 2D Convolution Shader
// Applies a 3x3 convolution kernel to a 2D image

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Input {
    float data[];
} input_image;

layout(set = 0, binding = 1) readonly buffer Kernel {
    float data[9];  // 3x3 kernel
} kernel;

layout(set = 0, binding = 2) writeonly buffer Output {
    float data[];
} output_image;

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
};

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    if (x >= width || y >= height) {
        return;
    }

    float sum = 0.0;

    // Apply 3x3 kernel
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = int(x) + kx;
            int py = int(y) + ky;

            // Clamp to image boundaries
            px = clamp(px, 0, int(width) - 1);
            py = clamp(py, 0, int(height) - 1);

            uint kernel_idx = uint((ky + 1) * 3 + (kx + 1));
            sum += input_image.data[py * width + px] * kernel.data[kernel_idx];
        }
    }

    output_image.data[y * width + x] = sum;
}
