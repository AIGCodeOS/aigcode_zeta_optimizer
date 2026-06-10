# Zeta Optimizer for Qwen3 Training (MindSpeed Integration)
## 1. 快速开始

### 脚本位置
所有启动脚本均位于 `examples/mcore/qwen3/` 目录下。请确保您的执行路径位于仓库根目录。

- **Qwen3 0.6B**: `./examples/mcore/qwen3/pretrain_qwen3_0point6b_4K_ptd_zeta.sh`
- **Qwen3 1.7B**: `./examples/mcore/qwen3/pretrain_qwen3_1point7b_4K_zeta.sh`
- **Qwen3 8B**: `./examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd_zeta.sh`

### 运行命令
```bash
# 以 0.6B 模型为例
bash examples/mcore/qwen3/pretrain_qwen3_0point6b_4K_ptd_zeta.sh <MASTER_PORT> <LR> <NPU_IDS>
```

## 2. 实验结果展示

### 性能表现总结
| 模型规模 | 优化器 | 序列长度 | 训练步数 | 收敛增益 (对比 AdamW) |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3 0.6B | Zeta | 4096 | 20,000 | ~15% Faster |
| Qwen3 1.7B | Zeta | 4096 | 20,000 | ~12% Faster |
| Qwen3 8B | Zeta | 4096 | 40,000 | TBD |

### 实验结果详细报告 (PDF)
由于当前环境限制，建议直接查看以下 PDF 报告以获取完整的 Loss 曲线和指标对比：

- **Qwen3 Moe 实验对比**: [loss_comparison_20260507_101546.pdf](assets/loss_comparison_20260507_101546.pdf)
- **Qwen3 1.7B 实验对比**: [loss_comparison_20260507_101803.pdf](assets/loss_comparison_20260507_101803.pdf)
- **Qwen3 8B 实验对比**: [loss_comparison_20260507_101918.pdf](assets/loss_comparison_20260507_101918.pdf)
- **Qwen3 0.6B 实验对比**: [loss_comparison_20260507_103023.pdf](assets/loss_comparison_20260507_103023.pdf)

> **注**：在 GitHub/GitLab 等平台预览时，点击链接即可在线查看 PDF 详情。

## 3. 引用与支持
如果您在研究中使用了 Zeta 优化器或 MindSpeed，请参考相关的技术文档。

---
*Open Source with MindSpeed*
