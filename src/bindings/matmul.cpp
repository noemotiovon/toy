#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "../../include/common.h"

extern "C" {
    void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K);
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA device");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA device");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    // Create output tensor
    auto C = torch::zeros({M, N}, A.options());
    
    // Launch CUDA kernel
    launch_matmul(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}

PYBIND11_MODULE(kernels_cuda, m) {
    m.def("matmul", &matmul_cuda, "CUDA matrix multiplication");
}
