#version 450

// Softmax Activation Shader (per-row)
// Applies softmax to each row of a 2D matrix
// softmax(x)[i] = exp(x[i]) / sum(exp(x[j]) for all j)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Input {
    float data[];
} input_data;

layout(set = 0, binding = 1) writeonly buffer Output {
    float data[];
} output_data;

layout(push_constant) uniform PushConstants {
    uint rows;
    uint cols;
};

shared float shared_max;
shared float shared_sum;

void main() {
    uint row = gl_WorkGroupID.x;
    uint col = gl_LocalInvocationID.x;

    if (row >= rows) {
        return;
    }

    uint base_idx = row * cols;

    // Find maximum in row (for numerical stability)
    float local_max = -1e38;
    for (uint i = col; i < cols; i += gl_WorkGroupSize.x) {
        local_max = max(local_max, input_data.data[base_idx + i]);
    }

    // Reduce to find global max
    if (col == 0) {
        shared_max = local_max;
    }
    barrier();

    atomicMax(shared_max, local_max);
    barrier();

    float max_val = shared_max;

    // Compute exp(x - max) and sum
    float local_sum = 0.0;
    for (uint i = col; i < cols; i += gl_WorkGroupSize.x) {
        local_sum += exp(input_data.data[base_idx + i] - max_val);
    }

    // Reduce to find global sum
    if (col == 0) {
        shared_sum = 0.0;
    }
    barrier();

    atomicAdd(shared_sum, local_sum);
    barrier();

    float sum_exp = shared_sum;

    // Write normalized values
    for (uint i = col; i < cols; i += gl_WorkGroupSize.x) {
        output_data.data[base_idx + i] = exp(input_data.data[base_idx + i] - max_val) / sum_exp;
    }
}
