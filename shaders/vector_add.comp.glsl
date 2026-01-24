#version 450

// Vector Addition Shader
// Adds two vectors element-wise: C[i] = A[i] + B[i]

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer InputA {
    float data[];
} input_a;

layout(set = 0, binding = 1) readonly buffer InputB {
    float data[];
} input_b;

layout(set = 0, binding = 2) writeonly buffer Output {
    float data[];
} output_data;

layout(push_constant) uniform PushConstants {
    uint element_count;
};

void main() {
    uint idx = gl_GlobalInvocationID.x;

    if (idx >= element_count) {
        return;
    }

    output_data.data[idx] = input_a.data[idx] + input_b.data[idx];
}
