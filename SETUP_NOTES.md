# Setup Notes (DeblurDet)

## 已做修改

- pipeline/pipeline.py
  - 在 `DeblurDetPipeline.from_config()` 中新增 `score_thr` 和 `nms_thr` 参数。
  - 构建 `CEASCDetector` 时传入这两个阈值。

- pipeline/run_pipeline.py
  - 新增 CLI 参数 `--nms-thr`。
  - 将 `--score-thr` 和 `--nms-thr` 传入 `DeblurDetPipeline.from_config()`。

- pipeline/run_adaptive.py
  - 新增 CLI 参数 `--score-thr` 和 `--nms-thr`。
  - 将这两个阈值传入 `CEASCDetector`。

## 环境安装与验证步骤

1) 安装 OpenCV（提供 `cv2`）：
   ```bash
   pip install opencv-python
   # 服务器无图形界面时可用 headless 版本
   pip install opencv-python-headless
   ```

2) 安装 MMDetection 与 MMCV（版本需匹配）：
   ```bash
   pip install mmdet==2.24.1
   pip install -U mmcv-full==1.5.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
   ```
   如果你的 CUDA / PyTorch 版本不同，请替换为对应的 MMCV 下载链接。

3) 编译 CEASC 稀疏卷积算子：
   ```bash
   cd third_party/CEASC/Sparse_conv && python setup.py install
   ```

4) 运行自适应流水线（示例）：
   ```bash
   python pipeline/run_adaptive.py \
     --input /path/to/images_or_image.jpg \
     --output results/ \
     --deblur-checkpoint weights/deblur/DeepDeblur_GOPRO.pt \
     --det-config configs/ceasc_gfl_res18_visdrone.py \
     --det-checkpoint weights/detect/ceasc_visdrone.pth \
     --blur-threshold 100 \
     --score-thr 0.35 \
     --nms-thr 0.5 \
     --vis --report
   ```

5) 可选：检查权重 meta 信息以确认模型结构：
   ```bash
   python - <<'PY'
   import torch
   ckpt = torch.load('weights/detect/your_weight.pth', map_location='cpu')
   meta = ckpt.get('meta', {})
   print(meta.get('config', 'no config found')[:500])
   PY
   ```

## 备注

- `--input` 必须是实际存在的图像文件或目录路径，占位符会导致 0 张图片被处理。
- `--score-thr` 和 `--nms-thr` 现在会在标准与自适应流水线中生效。
