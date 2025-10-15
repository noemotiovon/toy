# CUDA Kernel, Triton, and TileLang Comparison Project

这是一个学习项目，用于对比三种不同的GPU计算技术：CUDA Kernel、Triton和TileLang。项目实现了矩阵加法、矩阵乘法和Softmax等基础算子，并提供了精度测试和性能测试功能。

## 项目结构

```
toy/
├── cuda_kernels/          # CUDA kernel实现
│   ├── matrix_add.cu     # 矩阵加法CUDA kernel
│   ├── matrix_mul.cu     # 矩阵乘法CUDA kernel
│   ├── softmax.cu        # Softmax CUDA kernel
│   └── wrapper.py        # Python包装器
├── triton_ops/           # Triton算子实现
│   ├── matrix_add.py     # 矩阵加法Triton实现
│   ├── matrix_mul.py     # 矩阵乘法Triton实现
│   ├── softmax.py        # Softmax Triton实现
│   └── __init__.py
├── tilelang_ops/         # TileLang算子实现
│   ├── matrix_add.py     # 矩阵加法TileLang实现
│   ├── matrix_mul.py     # 矩阵乘法TileLang实现
│   ├── softmax.py        # Softmax TileLang实现
│   └── __init__.py
├── pytorch_ops/          # PyTorch参考实现
│   ├── matrix_add.py     # 矩阵加法PyTorch实现
│   ├── matrix_mul.py     # 矩阵乘法PyTorch实现
│   ├── softmax.py        # Softmax PyTorch实现
│   └── __init__.py
├── tests/                # 测试模块
│   ├── accuracy_test.py  # 精度测试
│   ├── performance_test.py # 性能测试
│   ├── visualization.py  # 结果可视化
│   └── run_all_tests.py  # 测试运行器
├── requirements.txt      # 依赖包
├── setup.py             # 安装脚本
└── README.md            # 项目说明
```

## 技术介绍

### 1. CUDA Kernel

CUDA (Compute Unified Device Architecture) 是NVIDIA开发的并行计算平台和编程模型。CUDA Kernel是运行在GPU上的函数，具有以下特点：

**优势：**
- 直接控制GPU硬件资源
- 最高性能优化潜力
- 完全控制内存访问模式
- 支持复杂的并行算法

**劣势：**
- 编程复杂度高
- 需要深入了解GPU架构
- 调试困难
- 代码可移植性差

**适用场景：**
- 对性能要求极高的应用
- 需要复杂并行算法的场景
- 深度学习框架的底层实现

### 2. Triton

Triton是OpenAI开发的高级GPU编程语言，旨在简化GPU编程。它提供了Python-like的语法，同时保持高性能。

**优势：**
- 简洁的Python-like语法
- 自动优化内存访问
- 内置并行化支持
- 易于学习和使用
- 良好的性能表现

**劣势：**
- 相对较新的技术
- 生态系统不如CUDA成熟
- 某些高级优化需要深入理解

**适用场景：**
- 快速原型开发
- 机器学习算子实现
- 需要平衡性能和开发效率的场景

### 3. TileLang

TileLang是一个领域特定语言(DSL)，专门用于张量操作。它提供了高级抽象，自动处理优化和并行化。

**优势：**
- 高级抽象，易于使用
- 自动优化
- 跨平台支持
- 声明式编程

**劣势：**
- 相对较新的技术
- 学习资源有限
- 某些优化可能不如手写代码

**适用场景：**
- 张量计算密集型应用
- 需要快速开发的场景
- 跨平台部署需求

## 安装和使用

### 环境要求

- Python 3.8+
- CUDA 11.0+ (如果使用GPU)
- PyTorch 2.0+
- Triton 2.0+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

#### 运行所有测试
```bash
python tests/run_all_tests.py
```

#### 只运行精度测试
```bash
python tests/run_all_tests.py --accuracy-only
```

#### 只运行性能测试
```bash
python tests/run_all_tests.py --performance-only
```

#### 指定设备
```bash
python tests/run_all_tests.py --device cuda
```

### 单独运行测试

#### 精度测试
```bash
python tests/accuracy_test.py
```

#### 性能测试
```bash
python tests/performance_test.py
```

## 实现的功能

### 支持的算子

1. **矩阵加法 (Matrix Addition)**
   - 实现：`A + B = C`
   - 支持任意大小的矩阵

2. **矩阵乘法 (Matrix Multiplication)**
   - 实现：`A @ B = C`
   - 支持不同维度的矩阵乘法

3. **Softmax**
   - 实现：`softmax(x) = exp(x) / sum(exp(x))`
   - 支持数值稳定的实现

### 测试功能

1. **精度测试**
   - 以PyTorch实现为基准
   - 计算最大绝对误差
   - 验证数值正确性

2. **性能测试**
   - 测量执行时间
   - 统计平均时间和标准差
   - 支持不同矩阵大小

3. **结果可视化**
   - 生成性能对比图表
   - 创建精度对比报告
   - 自动生成总结报告

## 性能对比

### 理论性能特点

| 技术 | 开发效率 | 运行性能 | 学习曲线 | 调试难度 |
|------|----------|----------|----------|----------|
| CUDA Kernel | 低 | 最高 | 陡峭 | 困难 |
| Triton | 高 | 高 | 平缓 | 中等 |
| TileLang | 最高 | 中等 | 最平缓 | 容易 |

### 实际性能表现

运行测试后，结果将保存在`results/`目录中，包括：

- 性能对比图表
- 精度测试报告
- 总结报告

## 技术对比总结

### CUDA Kernel
- **最佳性能**：直接控制硬件，性能最优
- **开发成本高**：需要深入了解GPU架构
- **适用场景**：对性能要求极高的生产环境

### Triton
- **平衡选择**：良好的性能和开发效率平衡
- **易于使用**：Python-like语法，学习成本低
- **适用场景**：机器学习研究和快速原型开发

### TileLang
- **最高效率**：声明式编程，开发最快
- **自动优化**：编译器自动处理优化
- **适用场景**：快速开发和跨平台部署

## 学习建议

1. **初学者**：从PyTorch开始，理解基础概念
2. **进阶学习**：尝试Triton，体验GPU编程
3. **高级优化**：学习CUDA Kernel，掌握底层优化
4. **前沿技术**：探索TileLang等新兴技术

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

MIT License

## 参考资料

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Triton Documentation](https://triton-lang.org/)
- [TileLang Documentation](https://tilelang.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
