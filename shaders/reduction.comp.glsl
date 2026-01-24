#version 450

// Parallel Reduction Shader
// Computes the sum of all elements in the input array
// Uses shared memory for efficient reduction

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Input {
    float data[];
} input_data;

layout(set = 0, binding = 1) buffer Output {
    float data[];
} output_data;

layout(push_constant) uniform PushConstants {
    uint element_count;
};

shared float shared_data[256];

void main() {
    uint local_id = gl_LocalInvocationID.x;
    uint global_id = gl_GlobalInvocationID.x;
    uint group_id = gl_WorkGroupID.x;

    // Load data into shared memory
    if (global_id < element_count) {
        shared_data[local_id] = input_data.data[global_id];
    } else {
        shared_data[local_id] = 0.0;
    }
    barrier();

    // Parallel reduction in shared memory
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_data[local_id] += shared_data[local_id + stride];
        }
        barrier();
    }

    // Write result for this workgroup
    if (local_id == 0) {
        output_data.data[group_id] = shared_data[0];
    }
}
