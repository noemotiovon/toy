#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "../../include/common.h"

extern "C" {
    void launch_matrix_add(const float* A, const float* B, float* C, int M, int N);
}

torch::Tensor matrix_add_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA device");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA device");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");
    TORCH_CHECK(A.sizes() == B.sizes(), "Input tensors must have same shape");
    
    int M = A.size(0);
    int N = A.size(1);
    
    // Create output tensor
    auto C = torch::zeros_like(A);
    
    // Launch CUDA kernel
    launch_matrix_add(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N
    );
    
    return C;
}

PYBIND11_MODULE(kernels_cuda, m) {
    m.def("matrix_add", &matrix_add_cuda, "CUDA matrix addition");
}
