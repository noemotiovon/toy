#include "../../include/common.h"

__global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        // Find maximum value in the row for numerical stability
        float max_val = input[row * N];
        for (int j = 1; j < N; j++) {
            max_val = fmaxf(max_val, input[row * N + j]);
        }
        
        // Compute sum of exponentials
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += expf(input[row * N + j] - max_val);
        }
        
        // Compute softmax
        for (int j = 0; j < N; j++) {
            output[row * N + j] = expf(input[row * N + j] - max_val) / sum;
        }
    }
}

extern "C" {
    void launch_softmax(
        const float* input, float* output,
        int M, int N
    ) {
        dim3 block_size(256);
        int grid_size = (M + block_size.x - 1) / block_size.x;
        
        softmax_kernel<<<grid_size, block_size>>>(
            input, output, M, N
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
