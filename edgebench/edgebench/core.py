"""edgebench.core — 公共工具、成本模型、带宽拟合。

零第三方依赖。设计原则：
  * 一切都是 JSON，落盘到 runs/<时间戳>/，可断点续跑、可对比
  * 成本模型的公式与 D:\\deepseek\\costmodel\\costmodel.mjs 保持一致
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# 标称带宽（GB/s），用户可覆盖
NOMINAL_BW = {
    "rtx3090": 936,
    "rtx4090": 1008,
    "a100-40g": 1555,
    "a100-80g": 2039,
    "h100-sxm": 3350,
    "l40s": 864,
    "rtx4060ti": 288,
    "rtx5060-laptop": 448,
}

# 名字 -> 标称带宽（GB/s）。顺序要紧：更具体的放前面，"5060" 必须排在 "4060" 前
BW_BY_NAME = [
    ("h200", 4800), ("h100", 3350), ("a100-80", 2039), ("a100", 1555),
    ("l40s", 864), ("l40", 864), ("a40", 696), ("a10", 600), ("l4", 300),
    ("6000ada", 960), ("a6000", 768), ("t4", 320), ("v100", 900),
    ("5090", 1792), ("5080", 960), ("5070", 672), ("5060", 448),
    ("4090", 1008), ("4080", 736), ("4070", 504), ("4060", 288),
    ("3090", 936), ("3080", 760), ("3060", 360),
]


def guess_nominal_bw(env_path: str = "runs/env.json",
                     default: float = 1008.0) -> Tuple[float, str]:
    """从 env.json 的卡名里猜标称带宽，猜不到返回默认值（并说明是默认）。"""
    try:
        env = read_json(env_path)
        gpus = env.get("gpu", {}).get("gpus") or []
        name = (gpus[0].get("name", "") if gpus else "").lower().replace(" ", "")
    except Exception:
        return default, "默认值（没找到 %s）" % env_path
    for pat, bw in BW_BY_NAME:
        if pat in name:
            return float(bw), "按卡名匹配 %s" % pat
    return default, "默认值（卡名 %r 未收录）" % name[:32]


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("median of empty sequence")
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_cmd(cmd: Sequence[str], timeout: Optional[float] = None,
            cwd: Optional[str] = None, check: bool = True) -> Dict[str, Any]:
    """跑外部命令，返回 {rc, out, err, seconds}。不抛异常，由调用方判断。"""
    t0 = time.time()
    try:
        p = subprocess.run(
            list(cmd), cwd=cwd, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = 124, (e.stdout or ""), (e.stderr or "")
    except FileNotFoundError as e:
        rc, out, err = 127, "", str(e)
    d = {"cmd": list(cmd), "rc": rc, "out": out, "err": err,
         "seconds": round(time.time() - t0, 3)}
    if check and rc != 0:
        raise RuntimeError("command failed (%s): %s\n%s" % (rc, " ".join(cmd), err[-800:]))
    return d


def table(rows: List[List[str]], header: List[str]) -> str:
    """朴素文本表格，终端里够看。"""
    all_rows = [header] + rows
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(header))]
    out = []
    for i, r in enumerate(all_rows):
        cells = [str(c).ljust(widths[j]) for j, c in enumerate(r)]
        out.append("  ".join(cells))
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def eprint(*a: Any) -> None:
    print(*a, file=sys.stderr)


# --------------------------------------------------------------------------- #
# 成本模型（与 costmodel.mjs 同公式）
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    key: str
    label: str
    params_b: float
    active_b: float
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    # MLA 这类非标准注意力的每 token KV 字节（16bit），给 0 表示用通用公式
    kv_bytes_at_16: int = 0
    moe: bool = False


MODELS: Dict[str, ModelSpec] = {
    "qwen3-1.7b": ModelSpec("qwen3-1.7b", "Qwen3-1.7B (dense)", 1.7, 1.7, 28, 2048, 16, 8, 128),
    "qwen3-4b": ModelSpec("qwen3-4b", "Qwen3-4B (dense)", 4.0, 4.0, 36, 2560, 32, 8, 128),
    "olmoe-1b-7b": ModelSpec("olmoe-1b-7b", "OLMoE-1B-7B (MoE 6.9/1.3)", 6.9, 1.3, 16, 2048, 16, 16, 128, moe=True),
    "qwen3-30b-a3b": ModelSpec("qwen3-30b-a3b", "Qwen3-30B-A3B (MoE 30/3.3)", 30.5, 3.3, 48, 2048, 32, 4, 128, moe=True),
    "deepseek-v2-lite": ModelSpec("deepseek-v2-lite", "DeepSeek-V2-Lite (MoE 16/2.4, MLA)", 15.7, 2.4, 27, 2048, 16, 16, 128,
                                  kv_bytes_at_16=27 * 576 * 2, moe=True),
    "llama3-70b": ModelSpec("llama3-70b", "Llama-3-70B (dense)", 70, 70, 80, 8192, 64, 8, 128),
    "deepseek-v3": ModelSpec("deepseek-v3", "DeepSeek-V3 (MoE 671/37, MLA)", 671, 37, 61, 7168, 128, 128, 128,
                             kv_bytes_at_16=61 * 576 * 2, moe=True),
}


def kv_bytes_per_token_16(m: ModelSpec) -> int:
    if m.kv_bytes_at_16:
        return m.kv_bytes_at_16
    return m.layers * 2 * m.kv_heads * m.head_dim * 2


def weight_bytes(m: ModelSpec, w_bits: float) -> float:
    return m.active_b * 1e9 * (w_bits / 8.0)


def kv_bytes(m: ModelSpec, ctx: int, batch: int, kv_bits: float, k: int = 0) -> float:
    per_token = kv_bytes_per_token_16(m) * (kv_bits / 16.0)
    return batch * (ctx + k) * per_token


def step_flops(m: ModelSpec, ctx: int, batch: int, k: int = 0) -> float:
    positions = batch * (1 + k)
    return 2 * m.active_b * 1e9 * positions + 4 * m.layers * positions * ctx * m.hidden


@dataclass
class Predict:
    bytes_total: float
    flops: float
    t_mem: float
    t_cmp: float
    t_step: float
    ms_per_token: float
    tok_s: float
    bound: str


def predict(m: ModelSpec, ctx: int = 4096, batch: int = 1, w_bits: float = 4.5,
            kv_bits: float = 16, bw_gbs: float = 320, t0_ms: float = 1.5,
            tflops: float = 60, k: int = 0, alpha: float = 0.0) -> Predict:
    """单次 decode 的耗时预测。bw_gbs 用实测反解值，别用标称。"""
    b = weight_bytes(m, w_bits) + kv_bytes(m, ctx, batch, kv_bits, k)
    f = step_flops(m, ctx, batch, k)
    t_mem = b / (bw_gbs * 1e9)
    t_cmp = f / (tflops * 1e12)
    t_step = max(t_mem, t_cmp) + t0_ms / 1000.0
    ms_per_token = t_step / max(1, batch * (1 + alpha * k)) * 1000.0
    return Predict(
        bytes_total=b, flops=f, t_mem=t_mem, t_cmp=t_cmp, t_step=t_step,
        ms_per_token=ms_per_token,
        tok_s=1000.0 / ms_per_token if ms_per_token else float("inf"),
        bound="mem" if t_mem >= t_cmp else "compute",
    )


# --------------------------------------------------------------------------- #
# 从实测点反解 bwEff 与 t0
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    t0_ms: float
    bw_gbs: float
    bw_eff: float          # 相对 nominal
    r2: float
    points: List[Dict[str, Any]] = field(default_factory=list)


def fit_bandwidth(points: Iterable[Dict[str, Any]], nominal_bw: float = 1008.0) -> FitResult:
    """最小二乘拟合 ms_per_token = t0 + bytes / BW。

    points: [{"label": str, "bytes": float(每 token 搬运字节), "ms": float}, ...]
    至少 2 个点、且 bytes 要拉开差距，否则解不稳定。
    """
    pts = list(points)
    if len(pts) < 2:
        raise ValueError("至少要 2 个数据点才能拟合")
    xs = [p["bytes"] for p in pts]
    ys = [p["ms"] for p in pts]
    if max(xs) - min(xs) < 0.2 * max(xs):
        raise ValueError("各点字节数太接近（<20%），拟合不出斜率；换不同大小的模型")

    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    inter = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (inter + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # slope 单位 ms/byte；1/slope = byte/ms；×1000/1e9 换成 GB/s
    bw = (1.0 / slope) / 1e6 if slope > 0 else float("inf")
    out = FitResult(t0_ms=inter, bw_gbs=bw, bw_eff=bw / nominal_bw, r2=r2)
    for p in pts:
        pred = inter + slope * p["bytes"]
        out.points.append({**p, "pred_ms": pred, "err_pct": (pred - p["ms"]) / p["ms"] * 100})
    return out
