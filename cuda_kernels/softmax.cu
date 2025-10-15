#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void softmax_kernel(float* input, float* output, int M, int N) {
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
    void launch_softmax(float* input, float* output, int M, int N,
                       dim3 gridDim, dim3 blockDim) {
        softmax_kernel<<<gridDim, blockDim>>>(input, output, M, N);
    }
}
