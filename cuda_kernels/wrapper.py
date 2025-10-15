import torch
import numpy as np
from torch.utils.cpp_extension import load
import os

# Load CUDA extensions
current_dir = os.path.dirname(os.path.abspath(__file__))

try:
    cuda_ops = load(
        name="cuda_ops",
        sources=[
            os.path.join(current_dir, "matrix_add.cu"),
            os.path.join(current_dir, "matrix_mul.cu"),
            os.path.join(current_dir, "softmax.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False
    )
except Exception as e:
    print(f"Warning: Could not load CUDA extensions: {e}")
    cuda_ops = None

class CUDAOps:
    @staticmethod
    def matrix_add(A, B):
        """Matrix addition using CUDA kernel"""
        if cuda_ops is None:
            raise RuntimeError("CUDA extensions not available")
        
        M, N = A.shape
        C = torch.zeros_like(A)
        
        # Calculate grid and block dimensions
        block_size = 16
        grid_x = (N + block_size - 1) // block_size
        grid_y = (M + block_size - 1) // block_size
        
        cuda_ops.launch_matrix_add(
            A.contiguous().cuda(),
            B.contiguous().cuda(),
            C.cuda(),
            M, N,
            (grid_x, grid_y, 1),
            (block_size, block_size, 1)
        )
        
        return C
    
    @staticmethod
    def matrix_mul(A, B):
        """Matrix multiplication using CUDA kernel"""
        if cuda_ops is None:
            raise RuntimeError("CUDA extensions not available")
        
        M, K = A.shape
        _, N = B.shape
        C = torch.zeros(M, N, device=A.device, dtype=A.dtype)
        
        # Calculate grid and block dimensions
        block_size = 16
        grid_x = (N + block_size - 1) // block_size
        grid_y = (M + block_size - 1) // block_size
        
        cuda_ops.launch_matrix_mul(
            A.contiguous().cuda(),
            B.contiguous().cuda(),
            C.cuda(),
            M, N, K,
            (grid_x, grid_y, 1),
            (block_size, block_size, 1)
        )
        
        return C
    
    @staticmethod
    def softmax(x):
        """Softmax using CUDA kernel"""
        if cuda_ops is None:
            raise RuntimeError("CUDA extensions not available")
        
        M, N = x.shape
        output = torch.zeros_like(x)
        
        # Calculate grid and block dimensions
        block_size = 256
        grid_size = (M + block_size - 1) // block_size
        
        cuda_ops.launch_softmax(
            x.contiguous().cuda(),
            output.cuda(),
            M, N,
            (grid_size, 1, 1),
            (block_size, 1, 1)
        )
        
        return output
