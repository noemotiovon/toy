#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrix_add_kernel(float* A, float* B, float* C, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < N && idy < M) {
        int index = idy * N + idx;
        C[index] = A[index] + B[index];
    }
}

extern "C" {
    void launch_matrix_add(float* A, float* B, float* C, int M, int N, 
                          dim3 gridDim, dim3 blockDim) {
        matrix_add_kernel<<<gridDim, blockDim>>>(A, B, C, M, N);
    }
}
