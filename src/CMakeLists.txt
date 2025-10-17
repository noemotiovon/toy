cmake_minimum_required(VERSION 3.18)
project(cuda_kernels)

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# Set CUDA properties
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/bindings)

# CUDA kernel sources
set(CUDA_KERNELS
    cuda/matrix_add_kernel.cu
    cuda/matmul_kernel.cu
    cuda/softmax_kernel.cu
)

# Binding sources
set(BINDING_SOURCES
    bindings/cuda_kernels.cpp
)

# Create CUDA library
add_library(cuda_kernels_lib STATIC ${CUDA_KERNELS})
set_target_properties(cuda_kernels_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Create Python extension
pybind11_add_module(kernels_cuda ${BINDING_SOURCES})
target_link_libraries(kernels_cuda PRIVATE cuda_kernels_lib ${TORCH_LIBRARIES})

# Set properties
set_target_properties(kernels_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Compiler flags
target_compile_options(kernels_cuda PRIVATE -O3)
target_compile_options(cuda_kernels_lib PRIVATE -O3)

# CUDA flags
set_target_properties(kernels_cuda PROPERTIES
    CUDA_ARCHITECTURES "50;60;70;75;80;86"
)
