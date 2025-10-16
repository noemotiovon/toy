#include "../../include/common.h"

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

extern "C" {
    void launch_matmul(
        const float* A, const float* B, float* C,
        int M, int N, int K
    ) {
        dim3 block_size = kernels::get_optimal_block_size(M, N);
        dim3 grid_size = kernels::get_grid_size(M, N, block_size);
        
        matmul_kernel<<<grid_size, block_size>>>(
            A, B, C, M, N, K
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
