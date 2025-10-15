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
    row = tl.program_id(0)
    
    # Load row data
    row_start = row * stride_im
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_start + col_offsets * stride_in
    
    # Load with mask for boundary checking
    mask = col_offsets < N
    row_data = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Find maximum for numerical stability
    row_max = tl.max(row_data, axis=0)
    
    # Compute exponentials
    row_data_shifted = row_data - row_max
    row_exp = tl.exp(row_data_shifted)
    
    # Compute sum
    row_sum = tl.sum(row_exp, axis=0)
    
    # Compute softmax
    row_softmax = row_exp / row_sum
    
    # Store result
    output_ptrs = output_ptr + row_start + col_offsets * stride_on
    tl.store(output_ptrs, row_softmax, mask=mask)

def softmax(x):
    """Softmax using Triton"""
    M, N = x.shape
    output = torch.empty_like(x)
    
    grid = (M,)
    
    softmax_kernel[grid](
        x, output,
        M, N,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=1024,
    )
    
    return output
