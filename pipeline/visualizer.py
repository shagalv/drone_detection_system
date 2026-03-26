"""
pipeline/visualizer.py
======================
将去模糊图像和检测结果可视化。

输出三联图（原图 | 去模糊图 | 检测结果图）
"""

import cv2
import numpy as np
from typing import Optional, Tuple

# 每个类别对应颜色（BGR）
CLASS_COLORS = [
    (0,   128, 255),  # 行人     - 橙色
    (0,   200, 255),  # 人群     - 黄橙
    (0,   255, 128),  # 自行车   - 绿
    (255,  80,  80),  # 汽车     - 蓝
    (255, 160,  80),  # 面包车   - 浅蓝
    (255, 200,   0),  # 卡车     - 青色
    (180,   0, 255),  # 三轮车   - 紫
    (100,   0, 255),  # 遮阳三轮 - 深紫
    (255,   0, 180),  # 公交车   - 品红
    (0,   200, 200),  # 摩托车   - 青绿
]

CLASS_NAMES_ZH = [
    "行人", "人群", "自行车", "汽车",
    "面包车", "卡车", "三轮车", "遮阳三轮车",
    "公交车", "摩托车",
]


class Visualizer:
    """
    检测结果可视化工具。

    Args:
        font_scale:   字体大小
        line_thick:   边框线宽
        alpha:        半透明填充透明度（0=不填充，1=不透明）
        show_chinese: 是否显示中文类别名
    """

    def __init__(
        self,
        font_scale: float = 0.4,
        line_thick: int = 1,
        alpha: float = 0.15,
        show_chinese: bool = True,
    ):
        self.font_scale = font_scale
        self.line_thick = line_thick
        self.alpha = alpha
        self.show_chinese = show_chinese
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    # ──────────────────────────────────────────────────────
    # 主接口
    # ──────────────────────────────────────────────────────

    def draw(self, pipeline_result) -> np.ndarray:
        """
        生成三联对比图：
            [原始模糊图] | [去模糊图] | [检测结果图]

        Args:
            pipeline_result: PipelineResult 对象
        Returns:
            横向拼接的三联图，BGR numpy
        """
        blurry = pipeline_result.blurry_image
        sharp  = pipeline_result.sharp_image
        det    = pipeline_result.detection

        # 在去模糊图上绘制检测框
        det_img = self.draw_detections(sharp.copy(), det)

        # 添加标题栏
        blurry_titled = self._add_title(blurry, "原始模糊图像")
        sharp_titled  = self._add_title(sharp,  "去模糊结果 (DeepDeblur)")
        det_titled    = self._add_title(
            det_img,
            f"目标检测 (CEASC) | 共 {det.num_objects} 个目标"
        )

        # 统计信息覆盖
        info_img = self._add_stats(det_titled, pipeline_result)

        return np.hstack([blurry_titled, sharp_titled, info_img])

    def draw_detections(self, image: np.ndarray, detection) -> np.ndarray:
        """
        在图像上绘制所有检测框。

        Args:
            image:     BGR numpy，会被原地修改
            detection: DetectionResult
        Returns:
            绘制后的图像
        """
        overlay = image.copy()

        for box, score, label in zip(detection.boxes, detection.scores, detection.labels):
            x1, y1, x2, y2 = map(int, box)
            color = CLASS_COLORS[label % len(CLASS_COLORS)]
            name  = CLASS_NAMES_ZH[label] if self.show_chinese else detection.class_names[label]

            # 半透明填充
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0, image)

        for box, score, label in zip(detection.boxes, detection.scores, detection.labels):
            x1, y1, x2, y2 = map(int, box)
            color = CLASS_COLORS[label % len(CLASS_COLORS)]
            name  = CLASS_NAMES_ZH[label] if self.show_chinese else detection.class_names[label]

            # 边框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_thick)

            # 标签背景
            label_text = f"{name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, self.font, self.font_scale, 1)
            ty = max(y1 - 2, th + 2)
            cv2.rectangle(image, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), color, -1)
            cv2.putText(image, label_text, (x1 + 1, ty), self.font, self.font_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)

        return image

    # ──────────────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────────────

    def _add_title(self, image: np.ndarray, title: str, bar_h: int = 24) -> np.ndarray:
        """在图像上方添加深色标题栏。"""
        bar = np.zeros((bar_h, image.shape[1], 3), dtype=np.uint8)
        bar[:] = (40, 40, 40)
        cv2.putText(bar, title, (6, bar_h - 6), self.font, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        return np.vstack([bar, image])

    def _add_stats(self, image: np.ndarray, result, margin: int = 6) -> np.ndarray:
        """在图像右下角添加耗时统计。"""
        lines = [
            f"去模糊: {result.deblur_time*1000:.0f}ms",
            f"检 测: {result.detect_time*1000:.0f}ms",
            f"总计:  {result.total_time*1000:.0f}ms",
        ]
        h = image.shape[0]
        fh = 16
        for i, line in enumerate(reversed(lines)):
            y = h - margin - i * fh
            cv2.putText(image, line, (margin, y), self.font, 0.38,
                        (50, 50, 50), 3, cv2.LINE_AA)
            cv2.putText(image, line, (margin, y), self.font, 0.38,
                        (220, 220, 60), 1, cv2.LINE_AA)
        return image

    def save_comparison(
        self,
        pipeline_result,
        save_path: str,
        jpeg_quality: int = 95,
    ) -> None:
        """保存三联对比图。"""
        canvas = self.draw(pipeline_result)
        cv2.imwrite(save_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        print(f"✅ 对比图已保存: {save_path}")
