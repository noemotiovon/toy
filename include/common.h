#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <vector>
#include <memory>

// CUDA error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << " - Status: " << status << std::endl; \
            exit(1); \
        } \
    } while(0)

// Common utility functions
namespace kernels {
    // Get optimal block size for given problem size
    inline dim3 get_optimal_block_size(int rows, int cols, int max_threads = 256) {
        int block_x = std::min(cols, 16);
        int block_y = std::min(rows, 16);
        while (block_x * block_y > max_threads) {
            if (block_x > block_y) block_x /= 2;
            else block_y /= 2;
        }
        return dim3(block_x, block_y);
    }
    
    // Calculate grid size
    inline dim3 get_grid_size(int rows, int cols, dim3 block_size) {
        int grid_x = (cols + block_size.x - 1) / block_size.x;
        int grid_y = (rows + block_size.y - 1) / block_size.y;
        return dim3(grid_x, grid_y);
    }
    
    // Memory alignment helper
    inline int align_up(int size, int alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }
}
