#!/usr/bin/env python3
"""
pipeline/run_adaptive.py
========================
自适应流水线入口：自动检测图像是否模糊，按需执行去模糊。

特点:
  - 清晰图跳过去模糊，节省约 120ms/张
  - 支持输出模糊检测报告
  - 适合混合质量的无人机视频流场景

用法:
    python pipeline/run_adaptive.py \
        --input drone_images/ \
        --output results/ \
        --deblur-checkpoint weights/deblur/DeepDeblur_GOPRO.pt \
        --det-config configs/ceasc_gfl_res18_visdrone.py \
        --det-checkpoint weights/detect/ceasc_gfl_visdrone.pth \
        --blur-threshold 100 \
        --report
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from deblur import DeblurModel
from detection import CEASCDetector
from tools.blur_assessment import BlurAssessor, AdaptivePipeline
from pipeline.visualizer import Visualizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run_adaptive")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="自适应去模糊检测流水线")
    p.add_argument("--input",             required=True)
    p.add_argument("--output",            default="results/")
    p.add_argument("--deblur-checkpoint", default="weights/deblur/DeepDeblur_GOPRO.pt")
    p.add_argument("--det-config",        default="configs/ceasc_gfl_res18_visdrone.py")
    p.add_argument("--det-checkpoint",    default="weights/detect/ceasc_gfl_visdrone.pth")
    p.add_argument("--device",            default="auto")
    p.add_argument("--blur-threshold",    type=float, default=100.0,
                   help="Laplacian 方差阈值，低于此值才去模糊（越小越保守）")
    p.add_argument("--vis",    action="store_true")
    p.add_argument("--report", action="store_true", help="输出每张图的模糊评估报告")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化各模块
    deblur   = DeblurModel(checkpoint=args.deblur_checkpoint, device=args.device)
    detector = CEASCDetector(config=args.det_config, checkpoint=args.det_checkpoint, device=args.device)
    assessor = BlurAssessor(method="laplacian", threshold=args.blur_threshold)
    pipeline = AdaptivePipeline(deblur, detector, assessor)
    visualizer = Visualizer()

    # 收集图像
    input_path = Path(args.input)
    image_files = sorted([
        p for p in (input_path.rglob("*") if input_path.is_dir() else [input_path])
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    logger.info(f"共找到 {len(image_files)} 张图像")

    report_data = []

    for idx, img_path in enumerate(image_files, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = pipeline.run(image)
        det = result["detection"]
        assessment = result["assessment"]

        status = "🔄去模糊" if result["did_deblur"] else "⚡直接检测"
        logger.info(
            f"[{idx}/{len(image_files)}] {img_path.name} "
            f"| {status} | 模糊评分: {assessment['laplacian_var']:.1f} "
            f"({assessment['blur_level']}) | 目标: {det.num_objects}"
        )

        if args.vis:
            from pipeline.pipeline import PipelineResult
            pr = PipelineResult(
                blurry_image=image,
                sharp_image=result["sharp_image"],
                detection=det,
                deblur_time=result["deblur_ms"] / 1000,
                detect_time=result["detect_ms"] / 1000,
            )
            vis_img = visualizer.draw(pr)
            cv2.imwrite(str(output_dir / f"{img_path.stem}_result.jpg"), vis_img)

        if args.report:
            report_data.append({
                "image":       img_path.name,
                "blur_score":  assessment["laplacian_var"],
                "blur_level":  assessment["blur_level"],
                "did_deblur":  result["did_deblur"],
                "num_objects": det.num_objects,
                "deblur_ms":   result["deblur_ms"],
                "detect_ms":   result["detect_ms"],
            })

    # 汇总统计
    stats = pipeline.stats
    logger.info(f"\n{'='*55}")
    logger.info(f"  完成！共 {stats['total']} 张")
    logger.info(f"  执行去模糊: {stats['deblurred']} 张 ({stats['deblur_rate']})")
    logger.info(f"  跳过去模糊: {stats['skipped']} 张")
    logger.info(f"{'='*55}")

    if args.report:
        report_path = output_dir / "blur_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"stats": stats, "images": report_data}, f, ensure_ascii=False, indent=2)
        logger.info(f"模糊报告已保存: {report_path}")


if __name__ == "__main__":
    main()
