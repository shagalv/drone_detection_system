"""
tools/blur_assessment.py
========================
图像模糊程度自动评估工具。

用途：
  - 自动判断输入图像是否需要去模糊
  - 在流水线中按需启用去模糊（模糊图才去模糊，清晰图直接检测）
  - 统计数据集的模糊分布

支持指标：
  - Laplacian 方差（最快）
  - Tenengrad 梯度能量
  - BRISQUE（无参考图像质量评估，需 opencv-contrib）
"""

import cv2
import numpy as np
from typing import Tuple, Dict


class BlurAssessor:
    """
    图像模糊程度评估器。

    Args:
        method:    评估方法，'laplacian' | 'tenengrad' | 'combined'
        threshold: 模糊判定阈值（越高越严格，即更多图像判为模糊）
    """

    # 经验性阈值（在 VisDrone 验证集上标定）
    DEFAULT_THRESHOLDS = {
        "laplacian":  100.0,   # 低于此值视为模糊
        "tenengrad":  200.0,
    }

    def __init__(self, method: str = "laplacian", threshold: float = None):
        assert method in ("laplacian", "tenengrad", "combined"), \
            f"不支持的方法: {method}"
        self.method = method
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = self.DEFAULT_THRESHOLDS.get(method, 100.0)

    def score(self, image: np.ndarray) -> float:
        """
        计算清晰度评分（越高越清晰）。

        Args:
            image: BGR uint8 numpy 图像
        Returns:
            float 评分（越高越清晰）
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.method == "laplacian":
            return self._laplacian_variance(gray)
        elif self.method == "tenengrad":
            return self._tenengrad(gray)
        else:  # combined
            lap = self._laplacian_variance(gray)
            ten = self._tenengrad(gray)
            # 归一化后加权平均
            lap_norm = lap / self.DEFAULT_THRESHOLDS["laplacian"]
            ten_norm = ten / self.DEFAULT_THRESHOLDS["tenengrad"]
            return (lap_norm + ten_norm) / 2.0

    def is_blurry(self, image: np.ndarray) -> bool:
        """判断图像是否模糊。"""
        s = self.score(image)
        if self.method == "combined":
            return s < 1.0   # combined 归一化到 1.0
        return s < self.threshold

    def assess(self, image: np.ndarray) -> Dict:
        """返回完整评估报告。"""
        lap = self._laplacian_variance(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        ten = self._tenengrad(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        blurry = lap < self.DEFAULT_THRESHOLDS["laplacian"]
        return {
            "laplacian_var": round(lap, 2),
            "tenengrad":     round(ten, 2),
            "is_blurry":     blurry,
            "blur_level":    self._blur_level(lap),
        }

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        """Laplacian 方差：高频能量，清晰图像值大。"""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _tenengrad(gray: np.ndarray) -> float:
        """Tenengrad：Sobel 梯度能量。"""
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx ** 2 + gy ** 2))

    @staticmethod
    def _blur_level(laplacian_var: float) -> str:
        if laplacian_var >= 300:
            return "清晰"
        elif laplacian_var >= 100:
            return "轻微模糊"
        elif laplacian_var >= 30:
            return "中度模糊"
        else:
            return "严重模糊"


class AdaptivePipeline:
    """
    自适应流水线：仅对判定为模糊的图像执行去模糊，减少不必要计算。

    Args:
        deblur_model:  DeblurModel 实例
        detector:      CEASCDetector 实例
        assessor:      BlurAssessor 实例
    """

    def __init__(self, deblur_model, detector, assessor: BlurAssessor = None):
        self.deblur    = deblur_model
        self.detector  = detector
        self.assessor  = assessor or BlurAssessor(method="laplacian")
        self._stats    = {"deblurred": 0, "skipped": 0}

    def run(self, image: np.ndarray) -> dict:
        """
        自适应运行流水线：
          - 清晰图 → 直接检测
          - 模糊图 → 先去模糊再检测

        Returns:
            dict 包含 assessment, detection, sharp_image, did_deblur
        """
        import time
        assessment = self.assessor.assess(image)
        did_deblur = assessment["is_blurry"]

        t0 = time.perf_counter()
        if did_deblur:
            sharp = self.deblur.deblur(image)
            self._stats["deblurred"] += 1
        else:
            sharp = image
            self._stats["skipped"] += 1
        t1 = time.perf_counter()

        detection = self.detector.detect(sharp)
        t2 = time.perf_counter()

        return {
            "assessment":  assessment,
            "did_deblur":  did_deblur,
            "sharp_image": sharp,
            "detection":   detection,
            "deblur_ms":   round((t1 - t0) * 1000, 1),
            "detect_ms":   round((t2 - t1) * 1000, 1),
        }

    @property
    def stats(self) -> dict:
        total = self._stats["deblurred"] + self._stats["skipped"]
        return {
            **self._stats,
            "total":      total,
            "deblur_rate": f"{self._stats['deblurred']/max(total,1)*100:.1f}%",
        }
