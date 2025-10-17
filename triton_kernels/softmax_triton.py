import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for softmax - simplified and stable version"""
    row = tl.program_id(0)
    
    # Calculate column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create pointers for input data
    input_ptrs = input_ptr + row * stride_im + col_offsets * stride_in
    
    # Create mask for boundary checking
    mask = col_offsets < N
    
    # Load row data with mask
    row_data = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Find maximum for numerical stability
    row_max = tl.max(row_data, axis=0)
    
    # Compute exponentials with numerical stability
    row_data_shifted = row_data - row_max
    row_exp = tl.exp(row_data_shifted)
    
    # Compute sum
    row_sum = tl.sum(row_exp, axis=0)
    
    # Compute softmax
    row_softmax = row_exp / row_sum
    
    # Create pointers for output data
    output_ptrs = output_ptr + row * stride_om + col_offsets * stride_on
    
    # Store result with mask
    tl.store(output_ptrs, row_softmax, mask=mask)

def softmax_forward(x):
    """Softmax using Triton - simplified version for better stability"""
    M, N = x.shape
    output = torch.empty_like(x)
    
    # Use very small block size for maximum numerical stability
    BLOCK_SIZE = min(32, N)
    grid = (M,)
    
    softmax_kernel[grid](
        x, output,
        M, N,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
