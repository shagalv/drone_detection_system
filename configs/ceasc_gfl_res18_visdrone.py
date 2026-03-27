# configs/ceasc_gfl_res18_visdrone.py
# =====================================================
# CEASC GFL ResNet-18 配置（VisDrone 数据集）
# 基于 third_party/CEASC/configs/UAV/dynamic_gfl_res18_visdrone.py
# 本文件是接口配置，实际训练配置请参考 third_party/CEASC/configs/
# =====================================================

# 直接继承 CEASC 原始配置
_base_ = "../third_party/CEASC/configs/UAV/dynamic_gfl_res18_visdrone.py"

# ── 以下为针对去模糊输出的调整 ──────────────────────────

# 数据增强：由于输入已经过去模糊，可适当降低 ColorJitter 强度
# （去模糊后的图像颜色更稳定）
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# 测试配置（推理时使用）
# # test_cfg = dict(
#     nms_pre=3000,          # NMS 前保留候选框数
#     min_bbox_size=0,
#     score_thr=0.05,        # 低阈值保留更多候选
#     nms=dict(type="nms", iou_threshold=0.5),
#     max_per_img=3000,
# )
