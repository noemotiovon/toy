import torch
import tilelang as tl
import tilelang.language as T
from typing import Callable

# ==========================================================
# TileLang 矩阵乘法 kernel
# ==========================================================
@tl.jit
def tilelang_matrix_mul_kernel(
    M: int,
    N: int,
    K: int,
    dtype: str = "float32",
) -> "Callable":
    """
    TileLang matrix multiplication kernel: C = A @ B

    Args:
        M: Number of rows of A and C
        N: Number of columns of B and C
        K: Inner dimension (columns of A / rows of B)
        dtype: Data type for computation

    Returns:
        Compiled matrix multiplication kernel
    """

    TILE_M = 32
    TILE_N = 32
    TILE_K = 32
    TM = tl.cdiv(M, TILE_M)
    TN = tl.cdiv(N, TILE_N)
    TK = tl.cdiv(K, TILE_K)

    @T.prim_func
    def matrix_mul_main(
        A: T.Tensor([M, K], dtype),
        B: T.Tensor([K, N], dtype),
        C: T.Tensor([M, N], dtype),
    ):
        # Parallel over rows
        with T.Kernel(M, threads=128) as (i_m):
            # Allocate tile fragments
            a_tile = T.alloc_fragment([TILE_M, TILE_K], dtype)
            b_tile = T.alloc_fragment([TILE_K, TILE_N], dtype)
            c_tile = T.alloc_fragment([TILE_M, TILE_N], dtype)

            # Initialize C_tile
            for i, j in T.grid(TILE_M, TILE_N):
                c_tile[i, j] = 0.0

            # Tile loops
            for bm in T.Pipelined(0, TM):
                for bn in T.Pipelined(0, TN):
                    for bk in T.Pipelined(0, TK):
                        # Load tiles
                        for i, k in T.grid(TILE_M, TILE_K):
                            if bm*TILE_M+i < M and bk*TILE_K+k < K:
                                a_tile[i, k] = A[bm*TILE_M+i, bk*TILE_K+k]
                        for k, j in T.grid(TILE_K, TILE_N):
                            if bk*TILE_K+k < K and bn*TILE_N+j < N:
                                b_tile[k, j] = B[bk*TILE_K+k, bn*TILE_N+j]

                        # Compute multiplication
                        for i, j, k in T.grid(TILE_M, TILE_N, TILE_K):
                            if bm*TILE_M+i < M and bn*TILE_N+j < N and bk*TILE_K+k < K:
                                c_tile[i, j] += a_tile[i, k] * b_tile[k, j]

                    # Store result
                    for i, j in T.grid(TILE_M, TILE_N):
                        if bm*TILE_M+i < M and bn*TILE_N+j < N:
                            C[bm*TILE_M+i, bn*TILE_N+j] = c_tile[i, j]

    return matrix_mul_main


# ==========================================================
# 前向函数
# ==========================================================
def matmul_forward(a: torch.Tensor, b: torch.Tensor):
    """
    TileLang matrix multiplication

    Args:
        a: Input tensor [M, K]
        b: Input tensor [K, N]

    Returns:
        C = a @ b
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"
    C = torch.empty((M, N), device=a.device, dtype=a.dtype)
    dtype = str(a.dtype).replace("torch.", "")
    kernel = tilelang_matrix_mul_kernel(M, N, K, dtype=dtype)
    kernel(a, b, C)
    return C


# ==========================================================
# 测试
# ==========================================================
if __name__ == "__main__":
    M, K, N = 64, 64, 64
    a = torch.ones((M, K), device="cuda", dtype=torch.float32)
    b = torch.full((K, N), 2.0, device="cuda", dtype=torch.float32)
    c = matrix_mul_forward(a, b)
    print("C[0,0] =", c[0, 0])  # 应输出 128.0
