#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "../../include/common.h"

extern "C" {
    void launch_softmax(const float* input, float* output, int M, int N);
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
    m.def("softmax", &softmax_cuda, "CUDA softmax");
}
