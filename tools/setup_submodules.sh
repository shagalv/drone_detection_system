#!/usr/bin/env bash
# =============================================================
# tools/setup_submodules.sh
# 自动克隆并配置 CEASC 和 DeepDeblur-PyTorch 子项目
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
THIRD_PARTY="$ROOT_DIR/third_party"

mkdir -p "$THIRD_PARTY"

echo "=========================================="
echo "  [1/2] 克隆 CEASC (无人机目标检测)"
echo "=========================================="
if [ ! -d "$THIRD_PARTY/CEASC" ]; then
    git clone https://github.com/Cuogeihong/CEASC.git "$THIRD_PARTY/CEASC"
    echo "✅ CEASC 克隆完成"
else
    echo "⏩ CEASC 已存在，跳过克隆"
fi

echo ""
echo "=========================================="
echo "  [2/2] 克隆 DeepDeblur-PyTorch (去模糊)"
echo "=========================================="
if [ ! -d "$THIRD_PARTY/DeepDeblur-PyTorch" ]; then
    git clone https://github.com/SeungjunNah/DeepDeblur-PyTorch.git "$THIRD_PARTY/DeepDeblur-PyTorch"
    echo "✅ DeepDeblur-PyTorch 克隆完成"
else
    echo "⏩ DeepDeblur-PyTorch 已存在，跳过克隆"
fi

echo ""
echo "=========================================="
echo "  安装 CEASC 稀疏卷积算子"
echo "=========================================="
cd "$THIRD_PARTY/CEASC/Sparse_conv"
python setup.py install
cd "$ROOT_DIR"
echo "✅ 稀疏卷积算子安装完成"

echo ""
echo "=========================================="
echo "  安装 CEASC 依赖"
echo "=========================================="
pip install nltk -q
pip install -r "$THIRD_PARTY/CEASC/requirements/albu.txt" -q
echo "✅ CEASC 依赖安装完成"

echo ""
echo "🎉 所有子模块配置完成！"
echo "下一步: bash tools/download_weights.sh"
