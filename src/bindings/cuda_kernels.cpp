#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "../../include/common.h"

extern "C" {
    void launch_matrix_add(const float* A, const float* B, float* C, int M, int N);
    void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K);
    void launch_softmax(const float* input, float* output, int M, int N);
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

torch::Tensor softmax_cuda(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");
    
    int M = input.size(0);
    int N = input.size(1);
    
    // Create output tensor
    auto output = torch::zeros_like(input);
    
    // Launch CUDA kernel
    launch_softmax(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );
    
    return output;
}

PYBIND11_MODULE(kernels_cuda, m) {
    m.doc() = "CUDA kernels for matrix operations";
    m.def("matrix_add", &matrix_add_cuda, "CUDA matrix addition");
    m.def("matmul", &matmul_cuda, "CUDA matrix multiplication");
    m.def("softmax", &softmax_cuda, "CUDA softmax");
}
