#include "../../include/common.h"

__global__ void matrix_add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B, 
    float* __restrict__ C,
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < N && idy < M) {
        int index = idy * N + idx;
        C[index] = A[index] + B[index];
    }
}

extern "C" {
    void launch_matrix_add(
        const float* A, const float* B, float* C,
        int M, int N
    ) {
        dim3 block_size = kernels::get_optimal_block_size(M, N);
        dim3 grid_size = kernels::get_grid_size(M, N, block_size);
        
        matrix_add_kernel<<<grid_size, block_size>>>(
            A, B, C, M, N
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
