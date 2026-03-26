#!/usr/bin/env bash
# =============================================================
# tools/download_weights.sh
# 下载去模糊和检测的预训练权重
# =============================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$ROOT_DIR/weights"

mkdir -p "$WEIGHTS_DIR/deblur"
mkdir -p "$WEIGHTS_DIR/detect"

echo "========================================"
echo "  下载 DeepDeblur GOPRO 预训练权重"
echo "========================================"
# DeepDeblur-PyTorch 的 GOPRO 模型（需自行替换为真实下载链接）
# 官方权重: https://github.com/SeungjunNah/DeepDeblur-PyTorch
DEBLUR_URL="https://drive.google.com/uc?id=1AfZhyUXEAMJtNZKE7f3Nk1EDm-1CqhGS"
if command -v gdown &>/dev/null; then
    gdown "$DEBLUR_URL" -O "$WEIGHTS_DIR/deblur/DeepDeblur_GOPRO.pt"
    echo "✅ DeepDeblur 权重下载完成"
else
    echo "⚠️  未检测到 gdown，请手动下载："
    echo "   pip install gdown"
    echo "   gdown \"$DEBLUR_URL\" -O $WEIGHTS_DIR/deblur/DeepDeblur_GOPRO.pt"
    echo ""
    echo "   或访问: https://github.com/SeungjunNah/DeepDeblur-PyTorch"
fi

echo ""
echo "========================================"
echo "  下载 CEASC VisDrone 预训练权重"
echo "========================================"
echo "请从 Google Drive 手动下载 CEASC 权重："
echo "  GFL CEASC: https://drive.google.com/drive/folders/1v7pby3LqmcIdDI52KKUQ43Ra3tBQdusR"
echo "  目标路径:  $WEIGHTS_DIR/detect/ceasc_gfl_visdrone.pth"
echo ""
echo "CEASC 权重下载完成后，请运行:"
echo "  python pipeline/run_pipeline.py --help"
