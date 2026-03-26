# 🚁 DeblurDet: 模糊无人机图像目标检测系统

> **先去模糊，再检测** — 将 [DeepDeblur](https://github.com/SeungjunNah/DeepDeblur-PyTorch)（动态场景去模糊）与 [CEASC](https://github.com/Cuogeihong/CEASC)（无人机目标检测）集成为端到端流水线。

---

## 系统架构

```
模糊无人机图像
      │
      ▼
┌─────────────────────────────────┐
│   DeepDeblur (去模糊模块)        │
│   Multi-scale CNN Deblurring    │
│   输出: 清晰重建图像              │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│   CEASC (目标检测模块)           │
│   Adaptive Sparse Conv + GFL   │
│   输出: 检测框 + 类别 + 置信度    │
└─────────────────────────────────┘
      │
      ▼
可视化结果 / JSON 输出
```

## 环境要求

- Python 3.8+
- PyTorch 1.10.1
- CUDA 11.1（推荐，也支持 CPU）
- mmdet == 2.24.1
- mmcv-full == 1.5.1

## 快速安装

```bash
# 1. 克隆本项目
git clone https://github.com/your_org/drone_detection_system.git
cd drone_detection_system

# 2. 安装依赖
pip install -r requirements.txt

# 3. 克隆子模块（DeepDeblur-PyTorch & CEASC）
bash tools/setup_submodules.sh

# 4. 编译 CEASC 稀疏卷积
cd third_party/CEASC/Sparse_conv
python setup.py install
cd ../../..

# 5. 下载预训练权重
bash tools/download_weights.sh
```

## 快速使用

### 单张图像推理

```bash
python pipeline/run_pipeline.py \
    --input  path/to/blurry_image.jpg \
    --output results/ \
    --deblur-checkpoint weights/deblur/DeepDeblur_GOPRO.pt \
    --det-config  configs/ceasc_gfl_res18_visdrone.py \
    --det-checkpoint weights/detect/ceasc_visdrone.pth \
    --vis
```

### 批量处理目录

```bash
python pipeline/run_pipeline.py \
    --input  path/to/blurry_folder/ \
    --output results/ \
    --batch-size 4 \
    --save-deblurred          # 同时保存去模糊后的中间结果
```

### 视频推理

```bash
python pipeline/run_video.py \
    --input  drone_video.mp4 \
    --output result_video.mp4 \
    --deblur-checkpoint weights/deblur/DeepDeblur_GOPRO.pt \
    --det-config  configs/ceasc_gfl_res18_visdrone.py \
    --det-checkpoint weights/detect/ceasc_visdrone.pth
```

## 模块说明

| 模块 | 路径 | 说明 |
|------|------|------|
| 去模糊封装 | `deblur/deblur_model.py` | DeepDeblur-PyTorch 推理接口 |
| 检测封装 | `detection/detector.py` | CEASC/MMDet 推理接口 |
| 流水线核心 | `pipeline/pipeline.py` | 串联去模糊与检测 |
| 主入口 | `pipeline/run_pipeline.py` | CLI 入口脚本 |
| 可视化 | `pipeline/visualizer.py` | 结果可视化工具 |
| 评估工具 | `tools/evaluate.py` | 在 VisDrone 上评估 mAP |

## VisDrone 数据集评估

```bash
python tools/evaluate.py \
    --data-root  datasets/VisDrone/ \
    --det-config  configs/ceasc_gfl_res18_visdrone.py \
    --det-checkpoint weights/detect/ceasc_visdrone.pth \
    --deblur-checkpoint weights/deblur/DeepDeblur_GOPRO.pt \
    --blur-severity 3     # 模拟模糊程度 1-5
```

## 引用

```bibtex
@misc{ceasc,
  title  = {Adaptive Sparse Convolutional Networks with Global Context Enhancement
             for Faster Object Detection on Drone Images},
  author = {Bowei Du and Yecheng Huang and Jiaxin Chen and Di Huang},
  year   = {2023},
  eprint = {2303.14488}
}

@InProceedings{Nah_2017_CVPR,
  author    = {Nah, Seungjun and Kim, Tae Hyun and Lee, Kyoung Mu},
  title     = {Deep Multi-Scale Convolutional Neural Network for Dynamic Scene Deblurring},
  booktitle = {CVPR},
  year      = {2017}
}
```
