#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""直测 MoE 路由热度：给 FusedMoE.select_experts 打补丁，统计每步每层被真正选中的专家并集。
用法: python router_probe.py --model <path> --batches 1,8,32 --max-tokens 64 --out runs-4090/router.json
"""
import argparse
import json
import os
import statistics
import time

# vLLM v1 默认把 engine 跑在子进程里，monkeypatch 传不过去；关掉多进程才能拿到路由数据
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V1", "1")

import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--batches", default="1,8,32")
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--max-len", type=int, default=4096)
ap.add_argument("--gpu-util", type=float, default=0.90)
ap.add_argument("--out", default="runs-4090/router.json")
args = ap.parse_args()

# ---- 打补丁：记录每次 select_experts 的 topk_ids ----
buf = []
raw = FusedMoE.__dict__["select_experts"]
fn = getattr(raw, "__func__", raw)


def wrapper(*a, **kw):
    out = fn(*a, **kw)
    try:
        topk_ids = out[1]
        if torch.is_tensor(topk_ids):
            buf.append(topk_ids.detach().to("cpu"))
    except Exception:
        pass
    return out


FusedMoE.select_experts = staticmethod(wrapper)

PROMPTS = [
    "Explain in plain words how a database index works.",
    "Write a short paragraph about why the sky looks blue.",
    "List three common mistakes in Python when handling files.",
    "Describe the difference between TCP and UDP.",
    "给出三条提高睡眠质量的建议。",
    "用一句话解释什么是梯度下降。",
    "Summarise the plot of a detective story in two sentences.",
    "What are the tradeoffs of using a cache?",
]

print("loading model ...", flush=True)
llm = LLM(
    model=args.model,
    max_model_len=args.max_len,
    gpu_memory_utilization=args.gpu_util,
    enforce_eager=True,   # CUDA graph 捕获期间不能做 CPU 拷贝，只能跑 eager
)
sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

rows = []
for b in [int(x) for x in args.batches.split(",")]:
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(b)]
    buf.clear()
    t0 = time.time()
    llm.generate(prompts, sp, use_tqdm=False)
    wall = time.time() - t0
    tok_s = (b * args.max_tokens) / wall

    # 只取 decode 步：每次调用的 token 数恰好等于 batch
    n_experts = 128
    per_layer = {}
    hist = [0] * n_experts
    for t in buf:
        n = t.shape[0]
        if n != b:
            continue
        ids = t.reshape(-1)
        per_layer.setdefault("union", []).append(torch.unique(ids).numel())
        bc = torch.bincount(ids, minlength=n_experts)[:n_experts]
        for i, c in enumerate(bc.tolist()):
            hist[i] += c
    unions = per_layer.get("union", [])
    union_frac = statistics.mean(unions) / n_experts if unions else 0.0
    top_share = 0.0
    if sum(hist):
        v = sorted(hist, reverse=True)
        top_share = sum(v[: max(1, len(v) // 8)]) / sum(v)   # 前 1/8 专家吃掉多少选择
    rows.append({
        "batch": b, "tok_s": round(tok_s, 1),
        "calls_total": len(buf), "steps_sampled": len(unions),
        "union_frac": round(union_frac, 4),
        "experts_per_layer": round(union_frac * n_experts, 1),
        "top_1_8_share": round(top_share, 4),
    })
    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    json.dump({"model": args.model, "rows": rows}, f, ensure_ascii=False, indent=2)
print("saved", args.out)
