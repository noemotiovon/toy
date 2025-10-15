# TileLang operator implementations
from .matrix_add import TileLangOps as MatrixAddOps
from .matrix_mul import TileLangOps as MatrixMulOps
from .softmax import TileLangOps as SoftmaxOps

def matrix_add(a, b):
    return MatrixAddOps.matrix_add(a, b)

def matrix_mul(a, b):
    return MatrixMulOps.matrix_mul(a, b)

def softmax(x):
    return SoftmaxOps.softmax(x)

__all__ = ['matrix_add', 'matrix_mul', 'softmax']