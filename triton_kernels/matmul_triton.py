import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Triton kernel for matrix multiplication - simplified and stable version"""
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32 for better precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduce along K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load A and B blocks
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Create masks for boundary checking
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load data with masks
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Use element-wise multiplication with manual reduction
        # This approach is more stable than tl.dot for certain configurations
        # a: (BLOCK_M, BLOCK_K), b: (BLOCK_K, BLOCK_N)
        # We compute: acc[m, n] += sum_k(a[m, k] * b[k, n])
        
        # Broadcast multiplication: (BLOCK_M, BLOCK_K, 1) * (1, BLOCK_K, BLOCK_N)
        # This creates (BLOCK_M, BLOCK_K, BLOCK_N) tensor
        a_expanded = a[:, :, None]  # (BLOCK_M, BLOCK_K, 1)
        b_expanded = b[None, :, :]  # (1, BLOCK_K, BLOCK_N)
        product = a_expanded * b_expanded  # (BLOCK_M, BLOCK_K, BLOCK_N)
        
        # Sum over K dimension
        acc += tl.sum(product, axis=1)

    # Write back result
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def matmul_forward(a, b):
    """Matrix multiplication using Triton"""
    M, K = a.shape
    _, N = b.shape
    
    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Use smaller block sizes for better numerical stability
    BLOCK_M, BLOCK_N, BLOCK_K = 8, 8, 8
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return c
