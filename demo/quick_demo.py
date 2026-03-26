#!/usr/bin/env python3
"""
demo/quick_demo.py
==================
快速演示：用随机噪声图展示流水线接口调用（不需要真实权重）。
正式使用时替换为真实模型。
"""

import sys
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from detection.detector import DetectionResult

# ── 模拟去模糊模型（输出原图 + 少量锐化） ────────────────


class MockDeblurModel:
    def deblur(self, image: np.ndarray) -> np.ndarray:
        # 模拟锐化效果
        kernel = np.array([[-1, -1, -1],
                            [-1,  9, -1],
                            [-1, -1, -1]])
        sharp = cv2.filter2D(image, -1, kernel)
        return np.clip(sharp.astype(int), 0, 255).astype(np.uint8)


# ── 模拟检测器（随机生成检测框） ──────────────────────────


class MockDetector:
    CLASS_NAMES = ["pedestrian", "car", "bicycle", "motor", "bus"]

    def detect(self, image: np.ndarray) -> DetectionResult:
        h, w = image.shape[:2]
        n = np.random.randint(3, 10)
        boxes = np.column_stack([
            np.random.randint(0, w // 2, n),
            np.random.randint(0, h // 2, n),
            np.random.randint(w // 2, w, n),
            np.random.randint(h // 2, h, n),
        ]).astype(np.float32)
        scores = np.random.uniform(0.4, 0.95, n).astype(np.float32)
        labels = np.random.randint(0, len(self.CLASS_NAMES), n)
        return DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=labels,
            class_names=self.CLASS_NAMES,
        )


def main():
    print("=" * 55)
    print("  DeblurDet 快速演示（使用模拟模型）")
    print("=" * 55)

    from pipeline.pipeline import DeblurDetPipeline, PipelineResult
    from pipeline.visualizer import Visualizer
    import time

    # 创建模拟图像（640x480 模拟无人机图）
    demo_img = np.random.randint(80, 200, (480, 640, 3), dtype=np.uint8)
    # 添加模糊模拟
    blurry = cv2.GaussianBlur(demo_img, (15, 15), 0)

    # 构建流水线（使用模拟组件）
    deblur   = MockDeblurModel()
    detector = MockDetector()
    pipeline = DeblurDetPipeline(deblur, detector)

    # 运行
    t0 = time.perf_counter()
    result = PipelineResult(
        blurry_image=blurry,
        sharp_image=deblur.deblur(blurry),
        detection=detector.detect(deblur.deblur(blurry)),
        deblur_time=0.12,
        detect_time=0.08,
    )
    print(f"\n✅ 流水线执行完成")
    print(f"   - 检测到目标数: {result.num_objects}")
    print(f"   - 去模糊耗时:  {result.deblur_time*1000:.0f}ms（模拟）")
    print(f"   - 检测耗时:    {result.detect_time*1000:.0f}ms（模拟）")
    print(f"   - 总耗时:      {result.total_time*1000:.0f}ms（模拟）")

    # 可视化
    vis = Visualizer()
    canvas = vis.draw(result)
    out_path = ROOT / "demo" / "demo_output.jpg"
    out_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"\n📷 演示输出已保存: {out_path}")
    print("\n正式使用请替换真实模型，运行:")
    print("  python pipeline/run_pipeline.py --help")


if __name__ == "__main__":
    main()
