"""edgebench.doctor — 兼容性体检。

在陌生机器（尤其租用云卡）上，先跑这个，再谈测量。它把"跑不起来"的常见原因
分成 OK / WARN / FAIL 三类，每条失败都给一条可执行的修法。
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import Any, Dict, List, Tuple

from .core import run_cmd, table

# 常见卡的算力等级；vLLM / torch 的预编译包必须覆盖这个 sm
KNOWN_SM = {
    "7.0": "V100",
    "7.5": "T4 / RTX 2080",
    "8.0": "A100 / A30",
    "8.6": "RTX 3090 / A10 / A40",
    "8.7": "Jetson Orin / IGX（嵌入式，vLLM 支持有限）",
    "8.9": "RTX 4090 / L40S / 6000 Ada",
    "9.0": "H100 / H200",
    "10.0": "B200",
    "12.0": "RTX 5090 / B40（需要 torch>=2.7、CUDA>=12.8）",
}

MIN_PY = (3, 9)


def _ok(name: str, detail: str = "") -> Dict[str, str]:
    return {"level": "OK", "name": name, "detail": detail}


def _warn(name: str, detail: str, fix: str = "") -> Dict[str, str]:
    return {"level": "WARN", "name": name, "detail": detail, "fix": fix}


def _fail(name: str, detail: str, fix: str = "") -> Dict[str, str]:
    return {"level": "FAIL", "name": name, "detail": detail, "fix": fix}


def check() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    # 1 Python
    if sys.version_info[:2] >= MIN_PY:
        out.append(_ok("python", sys.version.split()[0]))
    else:
        out.append(_fail("python", sys.version.split()[0], ">= 3.9；conda create -n eb python=3.11"))

    # 2 torch
    torch = None
    try:
        import torch  # noqa: F401
        out.append(_ok("torch", torch.__version__))
    except Exception as e:
        out.append(_fail("torch", str(e)[:80], "pip install torch --index-url https://download.pytorch.org/whl/cu121"))

    # 3 CUDA / 驱动 / 卡
    if torch is not None:
        if not torch.cuda.is_available():
            out.append(_fail("cuda", "torch.cuda.is_available() == False",
                             "驱动没装好或容器没挂 GPU：先确认 `nvidia-smi` 能出卡"))
        else:
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            sm = "%d.%d" % (major, minor)
            archs = torch.cuda.get_arch_list()
            tag = "sm_%d%d" % (major, minor)
            has_sass = tag in archs or any(a.startswith(tag) for a in archs)
            # 真相是能不能跑 kernel，不是 arch 列表里有没有：cu128 的 wheel 就不带 sm_89，
            # 4090 靠 sm_86 的 PTX 跑 JIT，实测能跑。所以这里真跑一次矩阵乘。
            try:
                _m = torch.randn(64, 64, device="cuda")
                (_m @ _m).sum().item()
                torch.cuda.synchronize()
                out.append(_ok("gpu", "%s (sm %s)" % (name, sm)))
                if not has_sass:
                    out.append(_warn(
                        "gpu-arch", "torch 的 arch 列表没有 sm_%s（%s）" % (sm, ", ".join(archs)),
                        "实测 kernel 能跑，是走 PTX JIT；首次调 kernel 慢几秒，不影响吞吐测量"))
            except Exception as e:
                out.append(_fail("gpu", "%s (sm %s) 跑不动 kernel：%s: %s"
                                       % (name, sm, type(e).__name__, str(e)[:120]),
                                 "换匹配 CUDA 版本的 torch wheel，或升级到支持 sm_%s 的版本" % sm))
            if sm not in KNOWN_SM:
                out.append(_warn("gpu", "sm %s 不在已知表里" % sm,
                                 "先看 torch.cuda.get_arch_list() 是否含 sm_%s%s；"
                                 "不含就必须换匹配的 wheel，否则一跑就报 no kernel image"
                                 % (major, minor)))
            if major < 8:
                out.append(_warn(
                    "gpu-capability",
                    "sm %s（Volta/Turing）：没有 BF16、没有 FlashAttention、没有 FP8" % sm,
                    "vLLM 必须 --dtype float16（本工具已自动切）；bf16 与 KV-fp8 两组实验在这张卡上"
                    "做不了，长上下文 prefill 也会明显慢"))

    # 4 vLLM
    try:
        import vllm
        ver = getattr(vllm, "__version__", "?")
        out.append(_ok("vllm", ver))
        try:
            mm = tuple(int(x) for x in ver.split(".")[:2])
            if mm < (0, 6):
                out.append(_warn("vllm", "%s 太老，continuous batching 行为不稳" % ver, "pip install -U vllm"))
            elif mm < (0, 8):
                out.append(_warn("vllm", "%s 不认识 Qwen3 架构" % ver, "跑 Qwen3 需要 >=0.8.4，否则换 Qwen2.5"))
        except Exception:
            pass
    except Exception:
        out.append(_warn("vllm", "没装", "bash setup.sh"))

    # 5 llama.cpp 二进制
    bin_dir = os.environ.get("LLAMA_BIN", "/root/llama.cpp/build/bin")
    for exe in ("llama-bench", "llama-cli", "llama-perplexity"):
        p = os.path.join(bin_dir, exe)
        if shutil.which(exe) or os.path.exists(p) or os.path.exists(p + ".exe"):
            out.append(_ok(exe, bin_dir))
        else:
            out.append(_warn(exe, "没找到（只有做 MoE 卸载和精度检查才需要）",
                             "bash setup.sh --with-llamacpp"))

    # 6 磁盘
    try:
        free = shutil.disk_usage(".").free / 1e9
        if free < 40:
            out.append(_warn("disk", "当前目录只剩 %.0f GB" % free,
                             "大模型要 30-80GB：换到 /root/autodl-tmp 再跑"))
        else:
            out.append(_ok("disk", "%.0f GB 可用" % free))
    except Exception as e:
        out.append(_warn("disk", str(e)[:60]))

    # 7 下载通道
    env = os.environ
    out.append(_ok("HF_ENDPOINT", env.get("HF_ENDPOINT", "(默认 huggingface.co)")))
    out.append(_ok("VLLM_USE_MODELSCOPE", env.get("VLLM_USE_MODELSCOPE", "(未设置)")))
    return out


def report(rows: List[Dict[str, str]]) -> str:
    lines = [table([[r["level"], r["name"], r["detail"][:60]] for r in rows],
                   ["级别", "检查项", "结果"])]
    fixes = [r for r in rows if r.get("fix") and r["level"] != "OK"]
    if fixes:
        lines.append("\n修法:")
        for r in fixes:
            lines.append("  [%s] %s: %s" % (r["level"], r["name"], r["fix"]))
    bad = sum(1 for r in rows if r["level"] == "FAIL")
    lines.append("\n结论: %s" % ("有 %d 项 FAIL，先修再跑测量" % bad if bad else "可以开跑"))
    return "\n".join(lines)


def main() -> int:
    rows = check()
    print(report(rows))
    return 1 if any(r["level"] == "FAIL" for r in rows) else 0
