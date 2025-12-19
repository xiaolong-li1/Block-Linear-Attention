# Block Linear Attention

基于 Triton 实现的稀疏线性注意力（Sparse Linear Attention）kernel。

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
    head_dim=64,      # 每个注意力头的维度
    topk=0.5,         # 选择 50% 的 KV blocks
    feature_map='softmax',  # 特征映射: 'softmax', 'elu', 'relu'
    BLKQ=64,          # Query block 大小
    BLKK=64,          # Key/Value block 大小
)

# 输入张量
B, H, L, D = 2, 8, 1024, 64  # batch, heads, seq_len, head_dim
q = torch.randn(B, H, L, D, device='cuda')
k = torch.randn(B, H, L, D, device='cuda')
v = torch.randn(B, H, L, D, device='cuda')

# 前向传播
output = attn(q, k, v)  # [B, H, L, D]

# 获取实际稀疏度
output, sparsity = attn(q, k, v, return_sparsity=True)
print(f"Actual sparsity: {sparsity:.2%}")
```

## API 说明

### SparseLinearAttention

主要的稀疏线性注意力模块。

```python
SparseLinearAttention(
    head_dim: int,           # 注意力头维度
    topk: float,             # 稀疏度 (0.0 ~ 1.0)，选择 topk 比例的 KV blocks
    feature_map: str = 'softmax',  # 特征映射函数
    BLKQ: int = 64,          # Query block 大小
    BLKK: int = 64,          # Key/Value block 大小
    use_bf16: bool = True,   # 使用 bfloat16 计算
    tie_feature_map_qk: bool = True,  # Q 和 K 使用相同的特征映射
)
```

**特征映射选项：**
- `'softmax'`: `softmax(x, dim=-1)`
- `'elu'`: `elu(x) + 1`
- `'relu'`: `relu(x)`

### SparseLinearAttentionWithProjection

带可学习投影层的版本，用于微调场景。

```python
from sparse_linear_attention import SparseLinearAttentionWithProjection

attn = SparseLinearAttentionWithProjection(
    head_dim=64,
    topk=0.5,
    feature_map='softmax',
)

output = attn(q, k, v)  # 自动应用残差投影
```

## 算法原理

### Linear Attention

标准线性注意力公式：

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
python -m sparse_linear_attention.test_kernel
```

测试内容：
- 前向传播正确性
- 反向传播正确性（梯度检查）
- 多种配置组合
- 数值梯度验证

## 文件结构

```
sparse_linear_attention/
├── __init__.py          # 模块导出
├── core.py              # SparseLinearAttention 模块
├── kernel.py            # Triton kernel 实现
├── utils.py             # 工具函数 (get_block_map, mean_pool)
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
