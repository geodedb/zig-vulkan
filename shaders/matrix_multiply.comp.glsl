#version 450

// Matrix Multiplication Shader
// C = A * B (naive implementation)
// A: M x K, B: K x N, C: M x N

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer MatrixA {
    float data[];
} mat_a;

layout(set = 0, binding = 1) readonly buffer MatrixB {
    float data[];
} mat_b;

layout(set = 0, binding = 2) writeonly buffer MatrixC {
    float data[];
} mat_c;

layout(push_constant) uniform PushConstants {
    uint M;  // rows of A, rows of C
    uint K;  // cols of A, rows of B
    uint N;  // cols of B, cols of C
};

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += mat_a.data[row * K + k] * mat_b.data[k * N + col];
    }

    mat_c.data[row * N + col] = sum;
}
