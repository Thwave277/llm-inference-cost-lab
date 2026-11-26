"""edgebench.bench — 两种引擎的吞吐测量。

vLLM 走离线 API，一次加载、多个 batch 点扫完；
llama.cpp 走 llama-bench，参数直通，便于做 MoE 卸载曲线。

计时口径（关键）：不信任引擎自报的数字，用"两个生成长度做差"来分离 prefill 与 decode：
    step_ms = (wall(长) - wall(短)) / (长 - 短)
这样加载时间、prefill、启动开销都被扣掉，剩下的就是纯 decode 每步耗时。
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence

from .core import run_cmd

DEFAULT_PROMPT = "用三百字解释：为什么大模型 decode 阶段受显存带宽而不是算力限制。"


# --------------------------------------------------------------------------- #
# vLLM
# --------------------------------------------------------------------------- #
def _auto_dtype() -> str:
    """sm < 8.0（V100 / T4 这类 Volta、Turing）没有 BF16，必须退回 FP16，
    否则模型 config 里的 bfloat16 会直接报 "not supported on this GPU"。"""
    try:
        import torch
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            return "float16" if major < 8 else "auto"
    except Exception:
        pass
    return "auto"


def vllm_sweep(model: str, batches: Sequence[int], max_tokens: int = 256,
               short_tokens: int = 8, kv_dtype: str = "auto", dtype: str = "auto",
               max_model_len: int = 4096, gpu_util: float = 0.90,
               tp: int = 1, prompt: str = DEFAULT_PROMPT,
               trust_remote_code: bool = False, enforce_eager: bool = False,
               prefix_caching: bool = False, ctx_tokens: int = 0,
               extra: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    from vllm import LLM  # 延迟导入

    if ctx_tokens:
        # 中文里一个字大约 1 token，按字数填到目标附近；真实 token 数加载后打印出来校准
        unit = ("推理系统的瓶颈在显存带宽：每生成一个 token，都要把全部权重从显存读一遍。"
                "这意味着批量越大，每 token 摊到的权重读取越少，吞吐越高。")
        prompt = (unit * (int(ctx_tokens / max(1, len(unit))) + 1))[:max(1, int(ctx_tokens))]

    if dtype == "auto":
        dtype = _auto_dtype()
    if kv_dtype == "fp8":
        try:
            import torch
            major, _ = torch.cuda.get_device_capability(0)
            if major < 8:
                raise RuntimeError(
                    "sm %d.x 不支持 FP8 KV cache（需要 sm89+），这条实验在 V100/T4 上跳过" % major)
        except ImportError:
            pass

    kwargs: Dict[str, Any] = dict(
        model=model, max_model_len=max_model_len, dtype=dtype,
        gpu_memory_utilization=gpu_util, tensor_parallel_size=tp,
        trust_remote_code=trust_remote_code, enforce_eager=enforce_eager,
        # 默认关掉前缀缓存：同一条 prompt 反复跑，缓存会让两次测量的起点不一致
        enable_prefix_caching=prefix_caching,
    )
    if kv_dtype and kv_dtype != "auto":
        kwargs["kv_cache_dtype"] = kv_dtype
    if extra:
        kwargs.update(extra)

    t_load = time.time()
    try:
        llm = LLM(**kwargs)
    except TypeError as e:            # 老版本 vLLM 没有这个参数
        if "enable_prefix_caching" in str(e):
            kwargs.pop("enable_prefix_caching", None)
            llm = LLM(**kwargs)
        else:
            raise
    load_s = time.time() - t_load

    if ctx_tokens:
        try:
            n_tok = len(llm.get_tokenizer().encode(prompt))
            print("  prompt 实际 %d tokens（目标 %d）" % (n_tok, ctx_tokens))
        except Exception:
            pass

    rows: List[Dict[str, Any]] = []
    for b in batches:
        try:
            row = _vllm_one(llm, model, b, max_tokens, short_tokens,
                            kv_dtype, dtype, load_s, prompt)
            if ctx_tokens:
                row["ctx_tokens"] = int(ctx_tokens)   # 让每次结果自带上下文长度，事后不用猜
            rows.append(row)
        except Exception as e:      # 单点 OOM 不该毁掉整轮
            msg = "%s: %s" % (type(e).__name__, str(e)[:200])
            print("  batch %-3d FAIL  %s" % (b, msg.splitlines()[0][:80]))
            rows.append({"engine": "vllm", "model": model, "batch": int(b),
                         "ok": False, "error": msg, "kv_dtype": kv_dtype})
    return rows


def _vllm_one(llm: Any, model: str, b: int, max_tokens: int, short_tokens: int,
              kv_dtype: str, dtype: str, load_s: float, prompt: str) -> Dict[str, Any]:
    prompts = [prompt] * int(b)
    # 短跑：主要成本是 prefill
    sp_short = _sampling(short_tokens)
    t0 = time.time()
    llm.generate(prompts, sp_short, use_tqdm=False)
    w_short = time.time() - t0
    # 长跑：prefill + 完整的 decode
    sp_long = _sampling(max_tokens)
    t0 = time.time()
    llm.generate(prompts, sp_long, use_tqdm=False)
    w_long = time.time() - t0

    steps = max_tokens - short_tokens
    step_ms = (w_long - w_short) / steps * 1000.0
    if step_ms <= 0:
        raise RuntimeError("step_ms 为负（两次测量抖动过大），加大 max_tokens 重跑")
    row = {
        "engine": "vllm", "model": model, "batch": int(b), "ok": True,
        "max_tokens": max_tokens, "wall_long_s": round(w_long, 3),
        "wall_short_s": round(w_short, 3),
        "step_ms": round(step_ms, 4),
        "ms_per_token": round(step_ms / b, 4),
        "tok_s": round(1000.0 / (step_ms / b), 1),
        "ttft_proxy_s": round(w_short, 3),
        "kv_dtype": kv_dtype, "dtype": dtype, "load_s": round(load_s, 1),
    }
    print("  batch %-3d step %7.2f ms  %6.1f tok/s  (load %.0fs)" % (
        b, step_ms, row["tok_s"], load_s))
    return row


def _sampling(max_tokens: int) -> Any:
    """贪心 + 不过早停，保证生成长度可控。"""
    from vllm import SamplingParams
    return SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)


# --------------------------------------------------------------------------- #
# llama.cpp
# --------------------------------------------------------------------------- #
def llama_bench(bin_dir: str, model_path: str, n_prompt: int = 512, n_gen: int = 256,
                reps: int = 5, ngl: int = 99, n_cpu_moe: Optional[int] = None,
                threads: Optional[int] = None, ctx: int = 4096,
                flash_attn: Optional[str] = None,
                extra_args: Optional[Sequence[str]] = None,
                timeout: float = 1800) -> Dict[str, Any]:
    """跑一次 llama-bench -o json，返回解析后的字段。

    flash_attn 传 "on"/"off"/"auto"；不传就不加这个参数（各版本默认值不一样，
    擅自指定会让跨版本对比失真）。
    """
    exe = os.path.join(bin_dir, "llama-bench")
    if os.name == "nt":
        exe += ".exe"
    cmd = [exe, "-m", model_path, "-p", str(n_prompt), "-n", str(n_gen),
           "-r", str(reps), "-ngl", str(ngl), "-c", str(ctx), "-o", "json"]
    if flash_attn:
        cmd += ["-fa", str(flash_attn)]
    if n_cpu_moe is not None:
        cmd += ["--n-cpu-moe", str(n_cpu_moe)]
    if threads:
        cmd += ["-t", str(threads)]
    if extra_args:
        cmd += list(extra_args)

    r = run_cmd(cmd, timeout=timeout, check=False)
    if r["rc"] != 0:
        return {"ok": False, "cmd": r["cmd"], "rc": r["rc"], "err": r["err"][-1200:],
                "n_cpu_moe": n_cpu_moe}

    import json as _json
    try:
        data = _json.loads(r["out"][r["out"].index("["):])
    except Exception:
        return {"ok": False, "cmd": r["cmd"], "rc": 0, "err": "输出不是 JSON:\n" + r["out"][-600:]}

    out: Dict[str, Any] = {"ok": True, "n_cpu_moe": n_cpu_moe, "ngl": ngl,
                           "n_prompt": n_prompt, "n_gen": n_gen, "rows": data}
    for item in data:
        # llama-bench 一行只对应一种负载：有 prompt 无生成为 pp，无 prompt 有生成为 tg
        if not item.get("n_prompt") and item.get("n_gen"):
            out["gen_tok_s"] = item.get("avg_ts")
            out["step_ms"] = round(1000.0 / item["avg_ts"], 4) if item.get("avg_ts") else None
        elif item.get("n_prompt") and not item.get("n_gen"):
            out["prefill_tok_s"] = item.get("avg_ts")
    return out


def moe_offload_curve(bin_dir: str, model_path: str, points: Sequence[int],
                      n_gen: int = 128, reps: int = 3,
                      extra_args: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """扫 --n-cpu-moe：0 = 全部专家在显存，越大越多的专家被放到 CPU。"""
    rows: List[Dict[str, Any]] = []
    for n in points:
        print("  n_cpu_moe=%-3d ..." % n, end="", flush=True)
        r = llama_bench(bin_dir, model_path, n_prompt=0, n_gen=n_gen, reps=reps,
                        n_cpu_moe=n, extra_args=extra_args)
        if r.get("ok") and r.get("gen_tok_s"):
            print("  %.1f tok/s" % r["gen_tok_s"])
        else:
            print("  FAIL")
        rows.append(r)
    return rows
