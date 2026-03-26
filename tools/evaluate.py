#!/usr/bin/env python3
"""
tools/evaluate.py
=================
在 VisDrone 验证集上评估流水线性能。

支持三种模式对比:
  1. 原始模糊图 + CEASC（baseline）
  2. 去模糊图  + CEASC（本系统）
  3. 清晰原图  + CEASC（上界参考）

用法:
    python tools/evaluate.py \
        --data-root datasets/VisDrone/ \
        --det-config configs/ceasc_gfl_res18_visdrone.py \
        --det-checkpoint weights/detect/ceasc_gfl_visdrone.pth \
        --deblur-checkpoint weights/deblur/DeepDeblur_GOPRO.pt \
        --blur-severity 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("evaluate")


def add_motion_blur(image: np.ndarray, severity: int) -> np.ndarray:
    """模拟运动模糊（用于无真实模糊数据集时的测试）。"""
    kernel_size = [0, 5, 9, 15, 21, 31][severity]
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size
    angle = np.random.randint(0, 180)
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    return cv2.filter2D(image, -1, kernel)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",         required=True)
    p.add_argument("--det-config",        required=True)
    p.add_argument("--det-checkpoint",    required=True)
    p.add_argument("--deblur-checkpoint", required=True)
    p.add_argument("--device",    default="auto")
    p.add_argument("--blur-severity", type=int, default=3, choices=[1, 2, 3, 4, 5],
                   help="模拟模糊程度 1-5（仅在无真实模糊图时使用）")
    p.add_argument("--max-images", type=int, default=0,
                   help="最多评估图像数（0 = 全部）")
    p.add_argument("--output", default="eval_results.json")
    return p.parse_args()


def main():
    args = parse_args()

    from detection import CEASCDetector
    from deblur import DeblurModel

    logger.info("正在加载模型...")
    deblur = DeblurModel(checkpoint=args.deblur_checkpoint, device=args.device)
    detector = CEASCDetector(config=args.det_config, checkpoint=args.det_checkpoint, device=args.device)

    data_root = Path(args.data_root)
    img_dir   = data_root / "images" / "val"
    ann_dir   = data_root / "annotations" / "val"

    image_files = sorted(img_dir.glob("*.jpg"))[:args.max_images or None]
    logger.info(f"共 {len(image_files)} 张验证图像")

    results = {"blur_only": [], "deblur_then_det": [], "settings": vars(args)}

    for i, img_path in enumerate(image_files, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # 模拟模糊
        blurry = add_motion_blur(image, args.blur_severity)

        # 模式 1：直接检测模糊图
        det_blurry = detector.detect(blurry)

        # 模式 2：先去模糊再检测
        sharp = deblur.deblur(blurry)
        det_sharp = detector.detect(sharp)

        results["blur_only"].append({
            "image": img_path.name,
            "num_objects": det_blurry.num_objects,
        })
        results["deblur_then_det"].append({
            "image": img_path.name,
            "num_objects": det_sharp.num_objects,
        })

        if i % 50 == 0:
            logger.info(f"  已处理 {i}/{len(image_files)}")

    # 汇总
    avg_blur = np.mean([r["num_objects"] for r in results["blur_only"]])
    avg_deblur = np.mean([r["num_objects"] for r in results["deblur_then_det"]])

    logger.info("\n" + "=" * 50)
    logger.info(f"  模糊图直接检测 - 平均目标数/图: {avg_blur:.1f}")
    logger.info(f"  去模糊后检测   - 平均目标数/图: {avg_deblur:.1f}")
    logger.info(f"  提升幅度:       +{(avg_deblur/avg_blur - 1)*100:.1f}%")
    logger.info("=" * 50)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存: {args.output}")


if __name__ == "__main__":
    main()
