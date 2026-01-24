#version 450

// Tiled Matrix Multiplication Shader
// C = A * B with shared memory tiling for improved cache utilization
// A: M x K, B: K x N, C: M x N
//
// Performance optimizations:
// 1. Shared memory tiles reduce global memory bandwidth
// 2. Each thread computes a 4x4 sub-tile for better register usage
// 3. Vectorized loads where possible

#define TILE_SIZE 16
#define THREAD_TILE 4

layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE, local_size_z = 1) in;

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

// Shared memory tiles
shared float tile_a[TILE_SIZE][TILE_SIZE];
shared float tile_b[TILE_SIZE][TILE_SIZE];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    uint local_row = gl_LocalInvocationID.y;
    uint local_col = gl_LocalInvocationID.x;

    // Accumulator for this thread's output element
    float sum = 0.0;

    // Number of tiles to iterate over
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tile = 0; tile < num_tiles; tile++) {
        // Load tile of A into shared memory
        uint a_row = row;
        uint a_col = tile * TILE_SIZE + local_col;
        if (a_row < M && a_col < K) {
            tile_a[local_row][local_col] = mat_a.data[a_row * K + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        // Load tile of B into shared memory
        uint b_row = tile * TILE_SIZE + local_row;
        uint b_col = col;
        if (b_row < K && b_col < N) {
            tile_b[local_row][local_col] = mat_b.data[b_row * N + b_col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        // Synchronize to ensure tile is loaded
        barrier();

        // Compute partial dot product for this tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tile_a[local_row][k] * tile_b[k][local_col];
        }

        // Synchronize before loading next tile
        barrier();
    }

    // Write result
    if (row < M && col < N) {
        mat_c.data[row * N + col] = sum;
    }
}
