import torch
import tilelang as tl
import tilelang.language as T
from typing import Callable

# ==========================================================
# TileLang 矩阵加法 kernel
# ==========================================================
@tl.jit
def tilelang_matrix_add_kernel(
    M: int,
    N: int,
    dtype: str = "float32",
) -> "Callable":
    """
    TileLang matrix addition kernel: C = A + B

    Args:
        M: Number of rows
        N: Number of columns
        dtype: Data type for computation

    Returns:
        Compiled matrix addition kernel
    """

    TILE_M = 32
    TILE_N = 32
    TM = tl.cdiv(M, TILE_M)
    TN = tl.cdiv(N, TILE_N)

    @T.prim_func
    def matrix_add_main(
        A: T.Tensor([M, N], dtype),
        B: T.Tensor([M, N], dtype),
        C: T.Tensor([M, N], dtype),
    ):
        # Parallel over rows
        with T.Kernel(M, threads=128) as (i_m):
            # Allocate tile fragments
            a_tile = T.alloc_fragment([TILE_M, TILE_N], dtype)
            b_tile = T.alloc_fragment([TILE_M, TILE_N], dtype)
            c_tile = T.alloc_fragment([TILE_M, TILE_N], dtype)

            # Tile loop over columns
            for bm in T.Pipelined(0, TM):
                for bn in T.Pipelined(0, TN):
                    # Load tile
                    for i, j in T.grid(TILE_M, TILE_N):
                        if bm*TILE_M+i < M and bn*TILE_N+j < N:
                            a_tile[i, j] = A[bm*TILE_M+i, bn*TILE_N+j]
                            b_tile[i, j] = B[bm*TILE_M+i, bn*TILE_N+j]

                    # Compute addition
                    for i, j in T.grid(TILE_M, TILE_N):
                        if bm*TILE_M+i < M and bn*TILE_N+j < N:
                            c_tile[i, j] = a_tile[i, j] + b_tile[i, j]

                    # Store result
                    for i, j in T.grid(TILE_M, TILE_N):
                        if bm*TILE_M+i < M and bn*TILE_N+j < N:
                            C[bm*TILE_M+i, bn*TILE_N+j] = c_tile[i, j]

    return matrix_add_main


# ==========================================================
# 前向函数
# ==========================================================
def matrix_add_forward(a: torch.Tensor, b: torch.Tensor):
    """
    TileLang matrix addition

    Args:
        a: Input tensor [M, N]
        b: Input tensor [M, N]

    Returns:
        C = a + b
    """
    M, N = a.shape
    C = torch.empty_like(a)
    dtype = str(a.dtype).replace("torch.", "")
    kernel = tilelang_matrix_add_kernel(M, N, dtype=dtype)
    kernel(a, b, C)
    return C

