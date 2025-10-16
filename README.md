# 高性能算子开发与对比框架

一个融合了 CUDA、Triton、TileLang、多后端接口与 PyTorch 验证体系的高性能算子开发与对比框架。

## 🚀 特性

- **多后端支持**: CUDA、Triton、TileLang、PyTorch
- **统一接口**: 高层统一调用接口，支持动态后端切换
- **精度验证**: 自动精度对比和一致性检查
- **性能测试**: 详细的性能分析和可视化
- **可扩展性**: 模块化设计，易于添加新算子和后端

## 📁 项目结构

```
toy/
├── README.md
├── setup.py
├── requirements.txt
├── pyproject.toml
│
├── docs/                        # 文档与设计说明
├── scripts/                     # 辅助脚本
├── include/                     # 公共头文件
├── src/                         # C++/CUDA 源码
├── triton_kernels/              # Triton 算子
├── tilelang_kernels/            # TileLang 算子
├── torch_kernels/               # PyTorch 参考实现
├── core/                        # 高层统一接口
├── tests/                       # 测试框架
└── examples/                    # 示例脚本
```

## 🛠️ 安装

```bash
# 克隆仓库
git clone https://github.com/noemotiovon/toy.git
cd your-repo

# 安装依赖
pip install -r requirements.txt

# 构建CUDA扩展
python setup.py build_ext --inplace
```

## 🎯 快速开始

```python
import torch
from core.kernel_registry import get_kernel
from core.performance_profiler import measure_time
from core.accuracy_checker import compare_tensors

# 准备测试数据
A = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
B = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)

# 基准实现
baseline = get_kernel("torch")(A, B)

# 对比不同后端
for backend in ["cuda", "triton", "tilelang"]:
    fn = get_kernel(backend)
    out = fn(A, B)
    acc = compare_tensors(out, baseline)
    t = measure_time(fn, A, B)
    print(f"[{backend}] acc={acc}, time={t*1e3:.3f} ms")
```

## 📊 运行基准测试

```bash
# 运行完整基准测试
python scripts/run_benchmark.py

# 运行精度测试
python scripts/compare_accuracy.py

# 可视化结果
python scripts/visualize_results.py
```

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_matrix_add.py
```

## 📖 文档

- [设计文档](docs/design.md)
- [内核架构](docs/kernel_architecture.md)
- [基准测试策略](docs/benchmark_strategy.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
