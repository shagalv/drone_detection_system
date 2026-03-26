#!/usr/bin/env python3
"""
pipeline/run_video.py
=====================
对无人机视频逐帧进行去模糊 + 目标检测，输出带检测框的视频。

用法:
    python pipeline/run_video.py \
        --input drone_blur.mp4 \
        --output result.mp4 \
        --deblur-checkpoint weights/deblur/DeepDeblur_GOPRO.pt \
        --det-config configs/ceasc_gfl_res18_visdrone.py \
        --det-checkpoint weights/detect/ceasc_gfl_visdrone.pth \
        --fps 15
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.pipeline import DeblurDetPipeline
from pipeline.visualizer import Visualizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run_video")


def parse_args():
    p = argparse.ArgumentParser(description="无人机视频去模糊 + 目标检测")
    p.add_argument("--input",  required=True)
    p.add_argument("--output", default="result.mp4")
    p.add_argument("--deblur-checkpoint", default="weights/deblur/DeepDeblur_GOPRO.pt")
    p.add_argument("--det-config",        default="configs/ceasc_gfl_res18_visdrone.py")
    p.add_argument("--det-checkpoint",    default="weights/detect/ceasc_gfl_visdrone.pth")
    p.add_argument("--device",   default="auto")
    p.add_argument("--fps",      type=float, default=0,
                   help="输出 FPS（0 = 跟随原视频）")
    p.add_argument("--max-frames", type=int, default=0,
                   help="最多处理帧数（0 = 全部）")
    p.add_argument("--show-side-by-side", action="store_true",
                   help="输出视频包含 [原始 | 去模糊+检测] 双画面")
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {args.input}")
        sys.exit(1)

    src_fps   = cap.get(cv2.CAP_PROP_FPS)
    out_fps   = args.fps if args.fps > 0 else src_fps
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_width = width * 2 if args.show_side_by_side else width
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, out_fps, (out_width, height))

    logger.info(f"视频信息: {width}x{height} @ {src_fps:.1f}fps, 共 {total_frames} 帧")

    # 初始化流水线
    pipeline   = DeblurDetPipeline.from_config(
        deblur_checkpoint=args.deblur_checkpoint,
        det_config=args.det_config,
        det_checkpoint=args.det_checkpoint,
        device=args.device,
    )
    visualizer = Visualizer()

    frame_idx  = 0
    t_start    = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        result = pipeline.run(frame)
        det_img = visualizer.draw_detections(result.sharp_image.copy(), result.detection)

        if args.show_side_by_side:
            out_frame = cv2.hconcat([frame, det_img])
        else:
            out_frame = det_img

        writer.write(out_frame)
        frame_idx += 1

        elapsed = time.perf_counter() - t_start
        fps_real = frame_idx / elapsed if elapsed > 0 else 0

        if frame_idx % 10 == 0:
            logger.info(f"  帧 {frame_idx}/{total_frames} | 实时处理速度: {fps_real:.1f} fps")

    cap.release()
    writer.release()
    logger.info(f"\n✅ 视频处理完成: {args.output}（共 {frame_idx} 帧）")


if __name__ == "__main__":
    main()
