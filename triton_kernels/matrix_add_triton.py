import torch
import triton
import triton.language as tl

@triton.jit
def matrix_add_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """Triton kernel for matrix addition"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Calculate offsets for this block
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create 2D offsets
    offs_a = offs_am[:, None] * stride_am + offs_bn[None, :] * stride_an
    offs_b = offs_am[:, None] * stride_bm + offs_bn[None, :] * stride_bn
    offs_c = offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    
    a_ptrs = a_ptr + offs_a
    b_ptrs = b_ptr + offs_b
    c_ptrs = c_ptr + offs_c
    
    # Create mask for boundary checking
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    
    # Load data with mask
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    
    # Compute addition
    c = a + b
    
    # Store result with mask
    tl.store(c_ptrs, c, mask=mask)

def matrix_add_forward(a, b):
    """Matrix addition using Triton"""
    M, N = a.shape
    c = torch.empty_like(a)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    
    matrix_add_kernel[grid](
        a, b, c,
        M, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16,
    )
    
    return c

# === 测试 ===
if __name__ == "__main__":
    M, N = 128, 128
    a = np.random.rand(M, N).astype("float32")
    b = np.random.rand(M, N).astype("float32")

    c = matrix_add_forward(a, b)
    print("Result match:", np.allclose(c, a + b))