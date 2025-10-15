from setuptools import setup, find_packages

setup(
    name="cuda-triton-tilelang-comparison",
    version="0.1.0",
    description="A comparison project for CUDA kernels, Triton, and TileLang implementations",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "triton>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
    ],
    python_requires=">=3.8",
)
