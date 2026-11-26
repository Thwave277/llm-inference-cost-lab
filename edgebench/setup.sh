#!/usr/bin/env bash
# edgebench 环境准备（AutoDL / Linux，纯 ASCII）
# 用法: bash setup.sh [--with-llamacpp]
set -euo pipefail
cd "$(dirname "$0")"

echo "== python =="
python -V
pip -V

echo "== pip 升级 =="
python -m pip install -U pip wheel setuptools

echo "== 装 vLLM（会自带匹配的 torch，别自己先装 torch 以免版本打架）=="
python -m pip install "vllm>=0.6.0" || {
  echo "安装失败：如果是 torch 版本冲突，用官方 wheel 索引重试"
  echo '  python -m pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121'
  exit 1
}

echo "== 装下载工具（modelscope 国内直连，hf 做备份）=="
python -m pip install -U "huggingface_hub[cli]" modelscope

echo "== 建议的下载镜像（国内）=="
echo '  export HF_ENDPOINT=https://hf-mirror.com'

if [ "${1:-}" = "--with-llamacpp" ]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "!! 没有 nvcc，无法编 CUDA 版 llama.cpp。装 CUDA toolkit 或跳过这步"
    echo "   （跳过也能跑全部 vLLM 实验，只是没有 MoE 卸载和 perplexity）"
  else
    echo "== 编译 llama.cpp（CUDA 版本，约 10-20 分钟）=="
    cd /root
    if [ ! -d llama.cpp ]; then
      git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
    fi
    cd llama.cpp
    cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
    cmake --build build --config Release -j "$(nproc)"
    echo "二进制在 /root/llama.cpp/build/bin"
  fi
fi

echo "== 自检 =="
python -m edgebench --help >/dev/null && echo "edgebench OK"
python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || true
