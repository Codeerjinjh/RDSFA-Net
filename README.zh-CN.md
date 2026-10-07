<div align="center">

# RDSFA-Net

### 面向 RGB-IR 目标检测的循环双域对齐与退化引导融合网络

吕国华，金继豪，高永彪，翟懿

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

</div>

---

## 📣 更新

- **[2026-10]** 开源 RDSFA-Net 的代码、模型配置与训练脚本。

## ✨ 方法亮点

RDSFA-Net 采用显式的 **对齐 → 估计 → 恢复 → 融合** 流程：不再假设可见光与红外图像
已经像素级配准、也不假设两个模态质量一致，而是在融合之前先显式校正跨模态几何偏差并抑制
空间变化的模态退化。

- **ECMI（早期跨模态交互）**：在骨干网络第一个 stage 之后即交换空间注意力与通道亲和度，
  使两个分支后续所有 stage 都具备跨模态感知能力。
- **RDSFA（循环双域谱特征对齐）**：用三个空洞可变形分支预测多尺度偏移场，通过
  squeeze-and-excitation 自适应加权，并以双向对称采样让两个模态相互靠拢；对齐可迭代
  精化，并与跨尺度反馈耦合。
- **EDRF（增强退化引导恢复融合）**：逐像素估计五个退化因子，进行退化门控恢复，并把特征
  分解为共享/私有部分后由动态模态门自适应重组，避免退化信号污染融合特征。
- **CSFR（跨尺度特征精化）**：以由细到粗的方式把高分辨率融合结果作为先验传递到粗尺度。

<div align="center">
<img src="assets/overview.png" width="100%">
<p><em>RDSFA-Net 整体结构。双分支 HGNet 骨干并行提取 RGB 与 IR 特征，ECMI、RDSFA、
EDRF、CSFR 依次嵌入 align-estimate-recover-fuse 流程，最终由 RT-DETR 解码器输出检测结果。</em></p>
</div>

## 📊 主要结果

输入分辨率 640×640，单卡 NVIDIA A800，mAP 为 COCO 标准。

| 方法 | 骨干 | 输入 | LLVIP mAP50 | LLVIP mAP50:95 | M3FD mAP50 | M3FD mAP50:95 | DVTOD mAP50 |
|---|---|---|---|---|---|---|---|
| RT-DETR-Concat（基线） | HGNetv2 | RGB+IR | – | – | 87.4 | 56.8 | 82.0 |
| **RDSFA-Net（本文）** | HGNetv2 | RGB+IR | **97.9** | **66.9** | **91.8** | **62.0** | **85.6** |

| 模型 | 参数量 | FPS | M3FD mAP50 | M3FD mAP50:95 | DVTOD mAP50 |
|---|---|---|---|---|---|
| RT-DETR-Concat | 66.4 M | 41.2 | 87.4 | 56.8 | 82.0 |
| **RDSFA-Net（本文）** | 120.0 M | 32.6 | **91.8** | **62.0** | **85.6** |

相比直接拼接，M3FD mAP50 提升 **+4.4**，DVTOD mAP50 提升 **+3.6**，且仍保持实时推理。
提升最明显的 **未配准** DVTOD 基准，直接体现了双向对称对齐的作用。

## 🗂 目录结构

本仓库基于 [yolo-multimodal](https://github.com/Limiiiing/yolo-multimodal)（Ultralytics
8.3.13 的多模态分支）整理发布。

```
RDSFA-Net/
├── ultralytics/
│   └── nn/
│       ├── AddModules/RDSFA.py             # ★ ECMI / RDSFA / EDRF / CSFR 实现
│       └── tasks.py                        # ★ parse_model() 中的模块注册
├── improve_multimodal/rtdetr/
│   ├── rtdetr-l-midfusion-RDSFA-Pro.yaml   # ★ 论文模型（约 130M 参数，500 queries，8 层解码器）
│   └── rtdetr-l-midfusion-RDSFA.yaml       # 轻量版本（不含 ECMI / CSFR）
├── train.py / train_pro.py                 # 训练入口
├── val.py / val_pro.py                     # 验证入口
├── predict.py                              # 推理与可视化
├── datalvip.yaml / datam3fd.yaml / datadvtod.yaml
├── assets/overview.png
└── INTEGRATION.md                          # 如何把 RDSFA 迁移到自己的代码库
```

带 ★ 的文件是本文的贡献，其余来自上游开源项目，详见"致谢"。

## 🔧 环境安装

```bash
git clone https://github.com/<your-account>/RDSFA-Net.git
cd RDSFA-Net

conda create -n rdsfa python=3.8 -y
conda activate rdsfa

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e .
```

## 📁 数据集准备

训练使用三个 RGB-IR 基准：**LLVIP**（已配准、低照度行人）、**M3FD**（已配准、六类目标）
与 **DVTOD**（**未配准**、无人机视角）。

数据加载器只读取可见光图像列表，再通过 `ir_replace` 规则推导红外图像路径，因此两个模态
必须共享文件名。

```yaml
# datalvip.yaml
path: /path/to/LLVIP/             # 数据集根目录，按需修改
train: images/train
val: images/val
ir_replace: ['images', 'image']   # images/xxx.jpg -> image/xxx.jpg
nc: 1
names:
  0: person
```

DVTOD 的可见光/红外目录为 `vis1` / `ir1`：

```yaml
ir_replace: ['vis1', 'ir1']
nc: 3
names: {0: person, 1: car, 2: bicycle}
```

## 🚀 训练 / 验证 / 推理

```bash
# 训练（论文模型为 rtdetr-l-midfusion-RDSFA-Pro.yaml）
python train_pro.py --data llvip     # 或 m3fd / dvtod

# 多卡 DDP
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_pro.py --data m3fd

# 验证
python val_pro.py --data m3fd --weights runs/train/RDSFA-Pro/M3FD_Pro/weights/best.pt
```

```python
# 推理：只需给出可见光图像目录，红外图像会自动配对
from ultralytics import RTDETR

model = RTDETR("runs/train/RDSFA-Pro/M3FD_Pro/weights/best.pt")
model.predict(source="datasets/M3FD/images/val", save=True, conf=0.25, iou=0.5)
```

## 🧩 迁移到自己的代码库

若你已有多模态 Ultralytics 分支，只需拷贝 `ultralytics/nn/AddModules/RDSFA.py`，
在 `AddModules/__init__.py` 中加入 `from .RDSFA import *`，并在
`ultralytics/nn/tasks.py` 的 `parse_model()` 中注册三个模块（`RDRFusion/RDSFusion`、
`CrossModalInteraction`、`SplitRGB/SplitIR`），最后使用
`improve_multimodal/rtdetr/rtdetr-l-midfusion-RDSFA-Pro.yaml` 作为模型定义。
完整改动见 [`INTEGRATION.md`](INTEGRATION.md)。

## 📖 引用

```bibtex
@article{lv2026rdsfanet,
  title   = {RDSFA-Net: Recurrent Dual-Domain Alignment and Degradation-Guided Fusion for RGB-Infrared Object Detection},
  author  = {Lv, Guohua and Jin, Jihao and Gao, Yongbiao and Zhai, Yi},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2026}
}
```

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics)：RDSFA-Net 所基于的检测框架（AGPL-3.0）。
- [yolo-multimodal](https://github.com/Limiiiing/yolo-multimodal)（作者 Limiiiing）：
  将 Ultralytics 扩展为 RGB-IR 多模态输入的分支；本仓库的多模态数据加载、RT-DETR 融合配置
  以及对比融合模块（ICAFusion、CFT、PSFM、DEYOLO 等）均来自该项目。其原始说明保留为
  `README_yolo-multimodal.md`。
- [LLVIP](https://bupt-ai-cz.github.io/LLVIP/)、
  [M3FD](https://github.com/JinyuanLiu-CV/TarDAL)、
  [DVTOD](https://github.com/VTI-Yan/DVTOD)：本文使用的数据集。

## 📄 许可证

本项目沿用 Ultralytics 的 **AGPL-3.0** 许可证发布，任何形式的使用均需遵守该协议，
详见 [`LICENSE`](LICENSE)。