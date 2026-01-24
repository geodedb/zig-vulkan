#version 450

// ReLU Activation Shader
// Applies ReLU activation: output[i] = max(0, input[i])

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Input {
    float data[];
} input_data;

layout(set = 0, binding = 1) writeonly buffer Output {
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

    float val = input_data.data[idx];
    output_data.data[idx] = max(0.0, val);
}
