import os
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
import torch

def get_extensions():
    extensions = []
    
    # CUDA extensions
    cuda_sources = [
        "src/cuda/matrix_add_kernel.cu",
        "src/cuda/matmul_kernel.cu", 
        "src/cuda/softmax_kernel.cu",
        "src/bindings/matrix_add.cpp",
        "src/bindings/matmul.cpp",
        "src/bindings/softmax.cpp",
    ]
    
    # Filter existing files
    existing_sources = [src for src in cuda_sources if os.path.exists(src)]
    
    if existing_sources:
        extensions.append(
            Pybind11Extension(
                "kernels_cuda",
                existing_sources,
                include_dirs=[
                    pybind11.get_cmake_dir() + "/../../../include",
                    "include",
                    torch.utils.cpp_extension.include_paths(),
                ],
                libraries=["cudart"],
                library_dirs=[torch.utils.cpp_extension.library_paths()],
                language="c++",
                cxx_std=17,
                extra_compile_args={
                    "nvcc": ["-O3", "--use_fast_math", "--expt-relaxed-constexpr"],
                    "cxx": ["-O3"],
                },
            )
        )
    
    return extensions

setup(
    name="high-performance-kernels",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="High-performance kernel development and comparison framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.0.0", 
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "pytest>=7.0.0",
        "scipy>=1.9.0",
    ],
)
