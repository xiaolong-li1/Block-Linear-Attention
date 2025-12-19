# Block Linear Attention

基于 Triton 实现的稀疏线性注意力（Sparse Linear Attention）kernel，比 Flash Attention (SDPA) 快 2-3 倍。

## 性能对比

| Seq Len | SDPA | SLA (topk=0.5) | 加速比 |
|---------|------|----------------|--------|
| 8192 | 1.17ms | 0.64ms | **1.8x** |
| 16384 | 4.58ms | 2.04ms | **2.2x** |
| 32768 | 18.86ms | 7.49ms | **2.5x** |
| 50000 | 45.00ms | 16.57ms | **2.7x** |

*测试配置: B=1, H=8, D=128, dtype=bfloat16*

## 安装依赖

```bash
pip install torch triton
```

## 快速开始

```python
import torch
from sparse_linear_attention import SparseLinearAttention

# 创建模块
attn = SparseLinearAttention(
    topk=0.5,             # 选择 50% 的 KV blocks
    feature_map='softmax',  # 特征映射: 'softmax', 'elu', 'relu'
    BLKQ=64,              # Query block 大小
    BLKK=64,              # Key/Value block 大小
)

# 输入张量
B, H, L, D = 1, 8, 50000, 128  # batch, heads, seq_len, head_dim
q = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
k = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
v = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)

# 前向传播
output = attn(q, k, v)  # [B, H, L, D]

# 获取实际稀疏度
output, sparsity = attn(q, k, v, return_sparsity=True)
print(f"Actual sparsity: {sparsity:.2%}")
```

## API

### SparseLinearAttention

```python
SparseLinearAttention(
    topk: float,             # 稀疏度 (0.0 ~ 1.0)，选择 topk 比例的 KV blocks
    feature_map: str = 'softmax',  # 特征映射函数
    BLKQ: int = 64,          # Query block 大小
    BLKK: int = 64,          # Key/Value block 大小
    use_bf16: bool = True,   # 使用 bfloat16 计算
)
```

**特征映射选项：**
- `'softmax'`: `softmax(x, dim=-1)`
- `'elu'`: `elu(x) + 1`
- `'relu'`: `relu(x)`

## 算法原理

### Linear Attention

```
O = (φ(Q) @ (φ(K)^T @ V)) / (φ(Q) @ sum(φ(K)))
```

其中 `φ` 是特征映射函数。

### Block-Sparse 加速

1. **Block 划分**：将 Q 和 K 按 block 划分
2. **Block 选择**：对每个 Query block，选择 top-k 个最相关的 KV blocks
3. **预计算**：对每个 KV block 预计算 `h_j = K_j^T @ V_j` 和 `z_j = sum(K_j)`
4. **稀疏累加**：只对选中的 blocks 累加结果

## 运行测试

```bash
# 正确性测试
python -m sparse_linear_attention.test_kernel

# 性能测试
python benchmark.py
```

## 文件结构

```
sparse_linear_attention/
├── __init__.py          # 模块导出
├── core.py              # SparseLinearAttention 模块
├── kernel.py            # Triton kernel 实现
├── utils.py             # 工具函数 (get_block_map)
├── reference_pytorch.py # PyTorch 参考实现（用于测试）
└── test_kernel.py       # 测试套件
```

## 引用

```bibtex
@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention},
  author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}
```

## License

Apache License 2.0
