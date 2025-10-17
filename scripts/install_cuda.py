#!/usr/bin/env python3
"""
Install CUDA kernels to make them available system-wide
"""

import os
import sys
import shutil
from pathlib import Path

def install_cuda_kernels():
    """Install CUDA kernels to site-packages"""
    
    # Find the compiled module
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    # Look for the compiled .so file
    possible_paths = [
        project_root / "kernels_cuda.so",
        project_root / "build" / "kernels_cuda.so",
        Path.cwd() / "kernels_cuda.so",
    ]
    
    compiled_module = None
    for path in possible_paths:
        if path.exists():
            compiled_module = path
            break
    
    if not compiled_module:
        print("No compiled CUDA module found. Please run build_cuda.py first.")
        return False
    
    # Get site-packages directory
    import site
    site_packages = site.getsitepackages()[0]
    
    # Copy module to site-packages
    target_path = Path(site_packages) / "kernels_cuda.so"
    
    try:
        shutil.copy2(compiled_module, target_path)
        print(f"CUDA kernels installed to: {target_path}")
        return True
    except Exception as e:
        print(f"Failed to install CUDA kernels: {e}")
        return False

def test_installation():
    """Test if the installation works"""
    try:
        import kernels_cuda
        print("CUDA kernels imported successfully!")
        
        # Quick test
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            a = torch.randn(32, 32, device=device, dtype=torch.float32)
            b = torch.randn(32, 32, device=device, dtype=torch.float32)
            
            c = kernels_cuda.matrix_add(a, b)
            print("CUDA matrix add test: PASS")
            
            return True
        else:
            print("CUDA not available for testing")
            return True
            
    except ImportError as e:
        print(f"Failed to import CUDA kernels: {e}")
        return False
    except Exception as e:
        print(f"Error testing CUDA kernels: {e}")
        return False

if __name__ == "__main__":
    if install_cuda_kernels():
        test_installation()
    else:
        sys.exit(1)
