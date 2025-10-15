#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrix_mul_kernel(float* A, float* B, float* C, int M, int N, int K) {
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
    void launch_matrix_mul(float* A, float* B, float* C, int M, int N, int K,
                          dim3 gridDim, dim3 blockDim) {
        matrix_mul_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
}
