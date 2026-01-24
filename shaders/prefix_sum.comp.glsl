#version 450

// Parallel Prefix Sum (Scan) Shader
// Computes inclusive prefix sum: output[i] = sum(input[0..i])
// Uses Blelloch algorithm (up-sweep and down-sweep)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    float data[];
} buffer_data;

layout(push_constant) uniform PushConstants {
    uint element_count;
    uint stage;  // 0 = up-sweep, 1 = down-sweep
};

shared float shared_data[512];  // 2x local size for padding

void main() {
    uint local_id = gl_LocalInvocationID.x;
    uint global_id = gl_GlobalInvocationID.x;
    uint group_id = gl_WorkGroupID.x;
    uint n = min(256u, element_count - group_id * 256);

    // Load data
    if (global_id < element_count) {
        shared_data[local_id] = buffer_data.data[global_id];
    } else {
        shared_data[local_id] = 0.0;
    }
    barrier();

    if (stage == 0) {
        // Up-sweep (reduce) phase
        uint offset = 1;
        for (uint d = n >> 1; d > 0; d >>= 1) {
            barrier();
            if (local_id < d) {
                uint ai = offset * (2 * local_id + 1) - 1;
                uint bi = offset * (2 * local_id + 2) - 1;
                if (bi < 256) {
                    shared_data[bi] += shared_data[ai];
                }
            }
            offset <<= 1;
        }
    } else {
        // Down-sweep phase
        if (local_id == 0) {
            shared_data[n - 1] = 0.0;
        }

        for (uint d = 1; d < n; d <<= 1) {
            uint offset = n >> 1;
            barrier();
            if (local_id < d) {
                uint ai = offset * (2 * local_id + 1) - 1;
                uint bi = offset * (2 * local_id + 2) - 1;
                if (bi < 256) {
                    float temp = shared_data[ai];
                    shared_data[ai] = shared_data[bi];
                    shared_data[bi] += temp;
                }
            }
        }
    }

    barrier();

    // Write result
    if (global_id < element_count) {
        buffer_data.data[global_id] = shared_data[local_id];
    }
}
