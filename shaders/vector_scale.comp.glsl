#version 450

// Vector Scale Shader
// Scales a vector by a scalar: B[i] = A[i] * scalar

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Input {
    float data[];
} input_data;

layout(set = 0, binding = 1) writeonly buffer Output {
    float data[];
} output_data;

layout(push_constant) uniform PushConstants {
    uint element_count;
    float scalar;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (idx >= element_count) {
        return;
    }

    output_data.data[idx] = input_data.data[idx] * scalar;
}
