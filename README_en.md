# Zeta Optimizer for Qwen3 Training (MindSpeed Integration)

[English](README_en.md) | [中文](README.md)

## 1. Quick Start

### Script Location
All startup scripts are located in the `examples/mcore/qwen3/` directory. Please ensure your execution path is at the repository root.

- **Qwen3 0.6B**: `./examples/mcore/qwen3/pretrain_qwen3_0point6b_4K_ptd_zeta.sh`
- **Qwen3 1.7B**: `./examples/mcore/qwen3/pretrain_qwen3_1point7b_4K_zeta.sh`
- **Qwen3 8B**: `./examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd_zeta.sh`

### Running Commands
```bash
# Using the 0.6B model as an example
bash examples/mcore/qwen3/pretrain_qwen3_0point6b_4K_ptd_zeta.sh <MASTER_PORT> <LR> <NPU_IDS>
```

## 2. Experimental Results

### Performance Summary
| Model Size | Optimizer | Sequence Length | Training Steps | Convergence Gain (vs AdamW) |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3 1.7B | Zeta | 4096 | 20,000 | 1.64x |
| Qwen3 8B | Zeta | 4096 | 40,000 | 1.2x |
| Qwen3 Moe | Zeta | 4096 | 20,000 | 1.32x |

### Result Visualization
Below are the Loss convergence curves for different Qwen3 model scales using the Zeta optimizer:

#### Qwen3 0.6B Experiment Comparison
![Qwen3 0.6B Loss](assets/qwen3-0.6b_page-0001.jpg)

#### Qwen3 1.7B Experiment Comparison
![Qwen3 1.7B Loss](assets/qwen3-1.7b_page-0001.jpg)

#### Qwen3 8B Experiment Comparison
![Qwen3 8B Loss](assets/qwen3-8b_page-0001.jpg)

#### Qwen3 MoE 1.3B Experiment Comparison
![Qwen3 MoE 1.3B Loss](assets/qwen3-moe1.3b_page-0001.jpg)

> **Note**: Detailed numerical reports can be found in the original PDF files within the `assets/` directory.

## 3. Citation and Support
If you use the Zeta optimizer or MindSpeed in your research, please refer to the relevant technical documentation.

---
*Open Source with MindSpeed*
