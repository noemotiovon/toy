#!/usr/bin/env python3
"""
Build CUDA kernels using PyTorch's JIT compilation
"""

import os
import sys
import torch
from torch.utils.cpp_extension import load

def build_cuda_kernels():
    """Build CUDA kernels using PyTorch's load function"""
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Define source files
    cuda_sources = [
        os.path.join(project_root, "src", "cuda", "matrix_add_kernel.cu"),
        os.path.join(project_root, "src", "cuda", "matmul_kernel.cu"),
        os.path.join(project_root, "src", "cuda", "softmax_kernel.cu"),
    ]
    
    cpp_sources = [
        os.path.join(project_root, "src", "bindings", "cuda_kernels.cpp"),
    ]
    
    # Include directories
    include_dirs = [
        os.path.join(project_root, "include"),
        os.path.join(project_root, "src", "bindings"),
    ]
    
    print("Building CUDA kernels...")
    print(f"CUDA sources: {cuda_sources}")
    print(f"CPP sources: {cpp_sources}")
    print(f"Include dirs: {include_dirs}")
    
    try:
        # Set build directory to project root
        build_dir = os.path.join(project_root, "build")
        os.makedirs(build_dir, exist_ok=True)
        
        # Load the extension
        kernels_cuda = load(
            name="kernels_cuda",
            sources=cuda_sources + cpp_sources,
            extra_include_paths=include_dirs,
            verbose=True,
            with_cuda=True,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            build_directory=build_dir,
        )
        
        print("CUDA kernels built successfully!")
        return kernels_cuda
        
    except Exception as e:
        print(f"Failed to build CUDA kernels: {e}")
        return None

def test_cuda_kernels():
    """Test the built CUDA kernels"""
    try:
        import kernels_cuda
        
        print("Testing CUDA kernels...")
        
        # Test matrix addition
        if torch.cuda.is_available():
            device = torch.device("cuda")
            a = torch.randn(64, 64, device=device, dtype=torch.float32)
            b = torch.randn(64, 64, device=device, dtype=torch.float32)
            
            # Test matrix add
            c_add = kernels_cuda.matrix_add(a, b)
            c_ref = a + b
            print(f"Matrix add test: {'PASS' if torch.allclose(c_add, c_ref) else 'FAIL'}")
            
            # Test matrix multiplication
            c_matmul = kernels_cuda.matmul(a, b)
            c_ref_matmul = torch.matmul(a, b)
            print(f"Matrix mul test: {'PASS' if torch.allclose(c_matmul, c_ref_matmul, atol=1e-4) else 'FAIL'}")
            
            # Test softmax
            x = torch.randn(32, 128, device=device, dtype=torch.float32)
            c_softmax = kernels_cuda.softmax(x)
            c_ref_softmax = torch.softmax(x, dim=-1)
            print(f"Softmax test: {'PASS' if torch.allclose(c_softmax, c_ref_softmax, atol=1e-4) else 'FAIL'}")
            
        else:
            print("CUDA not available, skipping tests")
            
    except ImportError:
        print("CUDA kernels not available")
    except Exception as e:
        print(f"Error testing CUDA kernels: {e}")

if __name__ == "__main__":
    kernels_cuda = build_cuda_kernels()
    if kernels_cuda:
        test_cuda_kernels()
    else:
        print("Build failed")
        sys.exit(1)
