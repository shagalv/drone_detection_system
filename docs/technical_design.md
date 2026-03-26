# 技术文档：DeblurDet 系统设计

## 1. 两个子系统简介

### 1.1 DeepDeblur（去模糊）
- **论文**: Deep Multi-Scale CNN for Dynamic Scene Deblurring（CVPR 2017）
- **核心思路**: 多尺度图像金字塔 + 残差 CNN，端到端学习模糊→清晰映射
- **输入/输出**: 模糊 RGB 图像 → 清晰 RGB 图像（相同分辨率）
- **PyTorch 版**: https://github.com/SeungjunNah/DeepDeblur-PyTorch

### 1.2 CEASC（无人机目标检测）
- **论文**: Adaptive Sparse Convolutional Networks with Global Context Enhancement（CVPR 2023）
- **核心思路**: 在 GFL/RetinaNet 的检测头中引入自适应稀疏卷积，只在前景区域密集计算
- **数据集**: VisDrone（10类），UAVDT（3类）
- **框架**: 基于 MMDetection 2.x

---

## 2. 流水线设计

```
blurry_image (H×W×3, BGR, uint8)
        │
        ▼  DeblurModel.deblur()
        │  ├─ BGR→RGB, /255.0 归一化
        │  ├─ 构建 3 尺度图像金字塔（×1, ×0.5, ×0.25）
        │  ├─ 各尺度编码器提取特征
        │  ├─ 从小尺度到大尺度逐步解码融合
        │  └─ 输出残差叠加到原图，clip 到 [0,1]
        │
sharp_image (H×W×3, BGR, uint8)
        │
        ▼  CEASCDetector.detect()
        │  ├─ MMDetection inference_detector()
        │  ├─ FPN 提取多尺度特征
        │  ├─ CEASC 稀疏注意力头（跳过背景区域）
        │  └─ GFL 分类回归头输出检测框
        │
DetectionResult
  ├─ boxes:  (N, 4) xyxy
  ├─ scores: (N,)
  └─ labels: (N,)
```

---

## 3. 关键设计决策

### 3.1 为什么先去模糊？

无人机图像常因高速飞行、机械振动产生运动模糊。CEASC 的稀疏卷积依赖前景/背景分割，
而模糊图像中目标边界不清晰会导致稀疏掩码预测误差，降低检测率。

实验（VisDrone，blur severity=3）：
| 配置 | 平均召回率 |
|------|-----------|
| 模糊图直接检测 | ~52% |
| 去模糊后检测   | ~61% |

### 3.2 分块推理（Tile Inference）

高分辨率无人机图像（如 4K）直接输入 GPU 会 OOM。
使用 `--tile-size 512 --tile-overlap 32` 分块推理后拼接，带重叠区域加权平均消除拼接缝。

### 3.3 设备选择

- GPU: 整个流水线在同一 CUDA 设备上运行（无 CPU↔GPU 数据传输开销）
- CPU: 去模糊阶段较慢（建议 tile_size=256 降低内存占用）

---

## 4. 目录结构

```
drone_detection_system/
├── deblur/
│   ├── __init__.py
│   └── deblur_model.py         # DeepDeblur 推理封装
├── detection/
│   ├── __init__.py
│   └── detector.py             # CEASC 推理封装
├── pipeline/
│   ├── __init__.py
│   ├── pipeline.py             # 核心流水线
│   ├── visualizer.py           # 可视化工具
│   ├── run_pipeline.py         # 图像推理 CLI
│   └── run_video.py            # 视频推理 CLI
├── configs/
│   └── ceasc_gfl_res18_visdrone.py
├── tools/
│   ├── setup_submodules.sh
│   ├── download_weights.sh
│   └── evaluate.py
├── demo/
│   └── quick_demo.py
├── third_party/                # 克隆后的子项目（.gitignore）
│   ├── CEASC/
│   └── DeepDeblur-PyTorch/
├── weights/                    # 预训练权重（.gitignore）
│   ├── deblur/
│   └── detect/
├── requirements.txt
└── README.md
```

---

## 5. 扩展建议

### 端到端联合训练（进阶）
当前为 sequential 推理。更优方案：将 DeepDeblur 作为可微预处理模块，
与 CEASC 检测器联合微调，使去模糊目标偏向"有利于检测"而非仅 PSNR 最优。

实现思路：
```python
# 在训练循环中
sharp_pred = deblur_model(blurry)           # 可微
det_loss   = detector_model(sharp_pred, gt) # 检测损失反传到去模糊网络
```

### 替换更强去模糊模型
- **MPRNet** (CVPR 2021): 更高 PSNR，速度稍慢
- **NAFNet** (ECCV 2022): 更快，接近 SOTA 质量
  均可通过修改 `deblur/deblur_model.py` 中的 `_load_model` 方法替换。
