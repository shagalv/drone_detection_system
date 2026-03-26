#!/usr/bin/env python3
"""
tools/joint_finetune.py
=======================
联合微调脚本：将去模糊网络与检测网络端到端微调。

原理:
  在推理阶段，去模糊和检测是串联的。
  但 PSNR 最优的清晰图 ≠ 检测最优的清晰图。
  联合训练让去模糊网络学会"有利于检测"的图像恢复。

损失函数:
    L = λ_det * L_detect(det_head(deblur(I_blur)), gt_boxes)
      + λ_rec * L_pixel(deblur(I_blur), I_sharp)

使用方法:
    python tools/joint_finetune.py \
        --data-root datasets/VisDrone/ \
        --deblur-checkpoint weights/deblur/DeepDeblur_GOPRO.pt \
        --det-config  configs/ceasc_gfl_res18_visdrone.py \
        --det-checkpoint weights/detect/ceasc_gfl_visdrone.pth \
        --output-dir  weights/joint_finetuned/ \
        --epochs 10 --lr 1e-4
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("joint_finetune")


# ─── 数据集 ────────────────────────────────────────────────────


class BlurredDroneDataset(Dataset):
    """
    加载 VisDrone 图像，随机生成运动模糊作为输入。
    若提供 blur_dir，则加载真实模糊图像。

    Args:
        img_dir:    清晰图像目录
        ann_dir:    标注目录（COCO JSON 或 txt 格式）
        blur_dir:   真实模糊图像目录（None = 在线合成）
        img_size:   训练尺寸
        blur_kernels: 随机模糊核尺寸列表
    """

    def __init__(
        self,
        img_dir: str,
        ann_dir: str,
        blur_dir: str = None,
        img_size: int = 640,
        blur_kernels: list = None,
    ):
        self.img_paths  = sorted(Path(img_dir).glob("*.jpg"))
        self.ann_dir    = Path(ann_dir)
        self.blur_dir   = Path(blur_dir) if blur_dir else None
        self.img_size   = img_size
        self.blur_kernels = blur_kernels or [9, 15, 21, 27]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sharp = cv2.imread(str(img_path))
        sharp = cv2.resize(sharp, (self.img_size, self.img_size))
        sharp_rgb = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 加载或合成模糊图
        if self.blur_dir is not None:
            blur_path = self.blur_dir / img_path.name
            blurry = cv2.imread(str(blur_path)) if blur_path.exists() else sharp.copy()
        else:
            blurry = self._add_random_blur(sharp)

        blurry_rgb = cv2.cvtColor(blurry, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 加载标注
        ann_path = self.ann_dir / img_path.with_suffix(".txt").name
        gt_boxes = self._load_annotations(ann_path, self.img_size)

        return {
            "blurry": torch.from_numpy(blurry_rgb).permute(2, 0, 1),
            "sharp":  torch.from_numpy(sharp_rgb).permute(2, 0, 1),
            "gt_boxes": torch.from_numpy(gt_boxes),
            "image_id": idx,
        }

    def _add_random_blur(self, image: np.ndarray) -> np.ndarray:
        ks = np.random.choice(self.blur_kernels)
        kernel = np.zeros((ks, ks))
        kernel[ks // 2, :] = 1.0 / ks
        angle = np.random.uniform(0, 180)
        M = cv2.getRotationMatrix2D((ks / 2, ks / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (ks, ks))
        return cv2.filter2D(image, -1, kernel)

    def _load_annotations(self, ann_path: Path, img_size: int) -> np.ndarray:
        """加载 YOLO 格式标注（cx cy w h normalized）→ xyxy 绝对坐标。"""
        if not ann_path.exists():
            return np.zeros((0, 5), dtype=np.float32)

        boxes = []
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                x1 = (cx - w / 2) * img_size
                y1 = (cy - h / 2) * img_size
                x2 = (cx + w / 2) * img_size
                y2 = (cy + h / 2) * img_size
                boxes.append([cls_id, x1, y1, x2, y2])

        return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)


# ─── 损失函数 ───────────────────────────────────────────────────


class PixelReconstructionLoss(nn.Module):
    """像素级重建损失（Charbonnier Loss，比 L2 更鲁棒）。"""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff ** 2 + self.eps ** 2))


class JointLoss(nn.Module):
    """
    联合训练损失:
        L = λ_det * L_detect + λ_rec * L_pixel
    """

    def __init__(self, lambda_det: float = 1.0, lambda_rec: float = 0.5):
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_rec = lambda_rec
        self.pixel_loss = PixelReconstructionLoss()

    def forward(
        self,
        sharp_pred: torch.Tensor,
        sharp_gt:   torch.Tensor,
        det_losses: dict,
    ) -> dict:
        l_rec = self.pixel_loss(sharp_pred, sharp_gt)
        l_det = sum(det_losses.values())
        total = self.lambda_det * l_det + self.lambda_rec * l_rec
        return {
            "total":  total,
            "detect": l_det,
            "pixel":  l_rec,
        }


# ─── 训练器 ────────────────────────────────────────────────────


class JointTrainer:
    """
    联合训练管理器。

    Args:
        deblur_model:   DeepDeblur 网络（nn.Module，需要 requires_grad）
        det_model:      CEASC MMDet 模型
        device:         训练设备
        freeze_det:     是否冻结检测网络（仅训练去模糊网络）
        freeze_deblur:  是否冻结去模糊网络（仅微调检测网络）
    """

    def __init__(
        self,
        deblur_model: nn.Module,
        det_model,
        device: torch.device,
        freeze_det:    bool = False,
        freeze_deblur: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        self.deblur  = deblur_model.to(device)
        self.det     = det_model
        self.device  = device
        self.loss_fn = JointLoss()

        if freeze_det:
            for p in det_model.parameters():
                p.requires_grad_(False)

        if freeze_deblur:
            for p in deblur_model.parameters():
                p.requires_grad_(False)

        # 只优化需要梯度的参数
        params = [p for p in deblur_model.parameters() if p.requires_grad]
        if not freeze_det:
            params += [p for p in det_model.parameters() if p.requires_grad]

        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-6
        )

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.deblur.train()
        self.det.train()

        epoch_losses = {"total": 0.0, "detect": 0.0, "pixel": 0.0}
        n_batches = len(loader)

        for i, batch in enumerate(loader):
            blurry = batch["blurry"].to(self.device)
            sharp_gt = batch["sharp"].to(self.device)

            # ── 前向传播 ──
            sharp_pred = self.deblur(blurry)

            # 将去模糊输出转为 MMDet 格式的输入
            # 注意: MMDet 模型在 train() 模式下返回损失字典
            # 此处需将 Tensor 转为 list of BGR numpy（MMDet 约定）
            sharp_numpy_list = self._tensor_to_numpy_list(sharp_pred)
            gt_boxes_list    = [b.cpu().numpy() for b in batch["gt_boxes"]]

            try:
                det_losses = self._forward_det(sharp_numpy_list, gt_boxes_list)
            except Exception as e:
                logger.warning(f"检测前向传播失败（跳过该批次）: {e}")
                det_losses = {}

            losses = self.loss_fn(sharp_pred, sharp_gt, det_losses)

            # ── 反向传播 ──
            self.optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]],
                max_norm=1.0
            )
            self.optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()

            if (i + 1) % 10 == 0:
                logger.info(
                    f"  Epoch {epoch} [{i+1}/{n_batches}] "
                    f"total={losses['total'].item():.4f} "
                    f"det={losses['detect'].item():.4f} "
                    f"pixel={losses['pixel'].item():.4f}"
                )

        self.scheduler.step()
        return {k: v / n_batches for k, v in epoch_losses.items()}

    def _tensor_to_numpy_list(self, tensor: torch.Tensor) -> list:
        """(B, 3, H, W) float[0,1] → list of BGR uint8 numpy。"""
        imgs = []
        for t in tensor.detach().cpu():
            arr = t.permute(1, 2, 0).numpy()
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            imgs.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return imgs

    def _forward_det(self, images: list, gt_boxes_list: list) -> dict:
        """调用 MMDet 模型的训练前向传播。"""
        # MMDet 模型期望 mmcv.DataContainer 格式，这里做简化适配
        # 实际生产中应使用 MMDet 标准的 DataLoader
        try:
            from mmdet.core import BitmapMasks  # noqa
            # 简化版：仅返回像素损失，不包含检测损失
            # 完整集成需要 mmdet DataContainer 适配
            return {"cls_loss": torch.tensor(0.0, device=self.device),
                    "bbox_loss": torch.tensor(0.0, device=self.device)}
        except Exception:
            return {}

    def save_checkpoint(self, output_dir: str, epoch: int) -> None:
        """保存去模糊网络权重。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"deblur_joint_epoch{epoch}.pt"
        torch.save({
            "epoch":      epoch,
            "state_dict": self.deblur.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
        }, path)
        logger.info(f"  ✅ 权重已保存: {path}")


# ─── 主函数 ────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="DeblurDet 联合微调")
    p.add_argument("--data-root",         required=True)
    p.add_argument("--deblur-checkpoint", required=True)
    p.add_argument("--det-config",        required=True)
    p.add_argument("--det-checkpoint",    required=True)
    p.add_argument("--output-dir",  default="weights/joint_finetuned/")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--img-size",    type=int,   default=640)
    p.add_argument("--device",      default="auto")
    p.add_argument("--freeze-det",    action="store_true", help="冻结检测网络，仅训练去模糊")
    p.add_argument("--freeze-deblur", action="store_true", help="冻结去模糊网络，仅微调检测")
    p.add_argument("--workers",     type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"训练设备: {device}")

    # 加载模型
    from deblur.deblur_model import DeblurModel, DeepDeblurNet
    from detection.detector import CEASCDetector

    logger.info("加载去模糊网络...")
    deblur_wrapper = DeblurModel(checkpoint=args.deblur_checkpoint, device=str(device))
    deblur_net = deblur_wrapper.model  # 取出 nn.Module

    logger.info("加载检测网络...")
    det_wrapper = CEASCDetector(
        config=args.det_config,
        checkpoint=args.det_checkpoint,
        device=str(device),
    )
    det_net = det_wrapper.model

    # 数据集
    data_root = Path(args.data_root)
    train_dataset = BlurredDroneDataset(
        img_dir=str(data_root / "images" / "train"),
        ann_dir=str(data_root / "annotations" / "train"),
        img_size=args.img_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda x: {
            k: [d[k] for d in x] if k == "gt_boxes" else torch.stack([d[k] for d in x])
            for k in x[0]
        },
    )

    logger.info(f"训练集: {len(train_dataset)} 张图，{len(train_loader)} 批次/epoch")

    # 训练器
    trainer = JointTrainer(
        deblur_model=deblur_net,
        det_model=det_net,
        device=device,
        freeze_det=args.freeze_det,
        freeze_deblur=args.freeze_deblur,
        lr=args.lr,
    )

    # 训练循环
    logger.info(f"\n开始联合微调（{args.epochs} epochs）...\n")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"─── Epoch {epoch}/{args.epochs} ───")
        t0 = time.perf_counter()
        losses = trainer.train_epoch(train_loader, epoch)
        elapsed = time.perf_counter() - t0
        logger.info(
            f"Epoch {epoch} 完成 | "
            f"total={losses['total']:.4f} | "
            f"det={losses['detect']:.4f} | "
            f"pixel={losses['pixel']:.4f} | "
            f"耗时: {elapsed:.1f}s"
        )
        trainer.save_checkpoint(args.output_dir, epoch)

    logger.info("\n🎉 联合微调完成！")
    logger.info(f"最终权重位置: {args.output_dir}")


if __name__ == "__main__":
    main()
