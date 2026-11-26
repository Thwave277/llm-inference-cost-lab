"""edgebench CLI：python -m edgebench <cmd>"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from . import core
from .core import MODELS, append_jsonl, fit_bandwidth, predict, read_json, table, write_json


def _parse_ints(s: str) -> List[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def cmd_doctor(a: argparse.Namespace) -> int:
    from . import doctor
    return doctor.main()


def cmd_pull(a: argparse.Namespace) -> int:
    from . import pull
    dest = a.dest or os.path.join("models", a.repo.replace("/", "--"))
    return pull.main(a.repo, dest, source=a.source, dry_run=a.dry_run)


def cmd_env(a: argparse.Namespace) -> int:
    from . import env as envmod
    envmod.main(a.out)
    print()
    print(envmod.clock_hint())
    return 0


def cmd_predict(a: argparse.Namespace) -> int:
    m = MODELS[a.model]
    p = predict(m, ctx=a.ctx, batch=a.batch, w_bits=a.w_bits, kv_bits=a.kv_bits,
                bw_gbs=a.bw, t0_ms=a.t0, tflops=a.tflops)
    print("%s  激活 %.1fB / 总 %.1fB" % (m.label, m.active_b, m.params_b))
    print("bytes/token %.2f GB   flops %.1f GFLOP   瓶颈 %s" % (
        p.bytes_total / 1e9, p.flops / 1e9, p.bound))
    print("ms/token %.2f   tok/s %.1f   (bw %.0f GB/s, t0 %.2f ms)" % (
        p.ms_per_token, p.tok_s, a.bw, a.t0))
    return 0


def cmd_fit(a: argparse.Namespace) -> int:
    pts = read_json(a.points)
    if a.nominal_bw:
        bw_ref, src = float(a.nominal_bw), "命令行指定"
    else:
        bw_ref, src = core.guess_nominal_bw(a.env)
    res = fit_bandwidth(pts, nominal_bw=bw_ref)
    rows = [[p.get("label", "?"), "%.2f" % (p["bytes"] / 1e9), "%.3f" % p["ms"],
             "%.3f" % p["pred_ms"], "%+.1f%%" % p["err_pct"]] for p in res.points]
    print(table(rows, ["点", "bytes(GB)", "实测 ms", "拟合 ms", "误差"]))
    print()
    print("t0      %.2f ms" % res.t0_ms)
    print("BW      %.0f GB/s  (标称 %.0f 的 %.0f%%，依据: %s)" % (
        res.bw_gbs, bw_ref, res.bw_eff * 100, src))
    print("R^2     %.4f" % res.r2)
    if res.r2 < 0.9:
        print("\n警告: R^2 < 0.9，点不够散或测量噪声太大，这组 t0/BW 不可信")
    out = {"t0_ms": round(res.t0_ms, 3), "bw_gbs": round(res.bw_gbs, 1),
           "bw_eff": round(res.bw_eff, 3), "r2": round(res.r2, 4),
           "points": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()}
                      for p in res.points]}
    if a.out:
        write_json(a.out, out)
        print("\n-> %s" % a.out)
    return 0


def cmd_sweep(a: argparse.Namespace) -> int:
    from . import bench
    batches = _parse_ints(a.batches)
    if a.dry_run:
        print("会跑: vllm_sweep(model=%s, batches=%s, max_tokens=%d, kv=%s, tp=%d)" % (
            a.model, batches, a.max_tokens, a.kv_dtype, a.tp))
        return 0
    rows = bench.vllm_sweep(a.model, batches, max_tokens=a.max_tokens,
                            kv_dtype=a.kv_dtype, dtype=a.dtype, max_model_len=a.max_len,
                            gpu_util=a.gpu_util, tp=a.tp, enforce_eager=a.eager,
                            prefix_caching=a.prefix_caching, ctx_tokens=a.ctx_tokens)
    out = a.out or os.path.join("runs", "sweep-%s.jsonl" % core.stamp())
    for r in rows:
        append_jsonl(out, r)
    print("\n-> %s" % out)
    return 0


def cmd_llama(a: argparse.Namespace) -> int:
    from . import bench
    if a.dry_run:
        print("会跑: llama_bench(%s, %s, n_gen=%d, ngl=%d, n_cpu_moe=%s)" % (
            a.bin_dir, a.model, a.n_gen, a.ngl, a.n_cpu_moe))
        return 0
    r = bench.llama_bench(a.bin_dir, a.model, n_prompt=a.n_prompt, n_gen=a.n_gen,
                          reps=a.reps, ngl=a.ngl, n_cpu_moe=a.n_cpu_moe,
                          threads=a.threads, ctx=a.ctx,
                          flash_attn=a.flash_attn or None)
    print(json.dumps(r, ensure_ascii=False, indent=2)[:2000])
    if a.out:
        write_json(a.out, r)
    return 0 if r.get("ok") else 1


def cmd_offload(a: argparse.Namespace) -> int:
    from . import bench
    pts = _parse_ints(a.points)
    if a.dry_run:
        print("会跑: moe_offload_curve(%s, %s, %s)" % (a.bin_dir, a.model, pts))
        return 0
    rows = bench.moe_offload_curve(a.bin_dir, a.model, pts, n_gen=a.n_gen, reps=a.reps)
    out = a.out or os.path.join("runs", "offload-%s.json" % core.stamp())
    write_json(out, rows)
    good = [r for r in rows if r.get("ok") and r.get("gen_tok_s")]
    if good:
        best = max(good, key=lambda r: r["gen_tok_s"])
        print("\n最快: n_cpu_moe=%s  %.1f tok/s" % (best["n_cpu_moe"], best["gen_tok_s"]))
    print("-> %s" % out)
    return 0


def cmd_ppl(a: argparse.Namespace) -> int:
    from . import precision
    r = precision.perplexity(a.bin_dir, a.model, a.corpus, ctx=a.ctx, ngl=a.ngl)
    print(json.dumps(r, ensure_ascii=False))
    if a.out:
        write_json(a.out, r)
    return 0 if r["ok"] else 1


def cmd_tasks(a: argparse.Namespace) -> int:
    from . import precision
    tasks = precision.load_tasks(a.tasks)
    if a.engine == "vllm":
        r = precision.run_tasks_vllm(a.model, tasks, max_tokens=a.max_tokens,
                                     max_model_len=a.max_len if hasattr(a, "max_len") else 4096)
    else:
        r = precision.run_tasks_llamacpp(a.bin_dir, a.model, tasks, ngl=a.ngl,
                                         max_tokens=a.max_tokens)
    print("通过 %d/%d" % (r["passed"], r["total"]))
    for d in r["detail"]:
        print("  %s  %s" % ("PASS" if d["ok"] else "FAIL", d["prompt"]))
    if a.out:
        write_json(a.out, r)
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    """把目录里的 jsonl 汇总成一张表。"""
    if not os.path.isdir(a.dir):
        print("目录不存在: %s" % a.dir)
        return 1
    files = sorted(f for f in os.listdir(a.dir) if f.endswith(".jsonl"))
    if not files:
        print("目录里没有 jsonl: %s" % a.dir)
        return 1
    for f in files:
        rows = []
        with open(os.path.join(a.dir, f), encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        # 失败行（OOM 等）没有 step_ms，不能进表格，单独计数
        good = [r for r in rows if r.get("batch") is not None
                and r.get("ok", True) and r.get("step_ms") is not None]
        bad = len(rows) - len(good)
        if not good:
            print("\n%s  没有有效行（%d 条失败/不完整）" % (f, bad))
            continue
        keys = [k for k in ("batch", "step_ms", "ms_per_token", "tok_s", "kv_dtype")
                if any(k in r for r in good)]
        print("\n%s%s" % (f, "  (%d 条失败未计入)" % bad if bad else ""))
        print(table([[str(r.get(k, "")) for k in keys] for r in good], keys))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edgebench", description="消费级显卡 LLM 推理基准与成本校准")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("env", help="记录环境快照")
    s.add_argument("--out", default="runs/env.json")
    s.set_defaults(func=cmd_env)

    s = sub.add_parser("doctor", help="兼容性体检（换机器先跑这个）")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("pull", help="下模型（modelscope / hf-mirror 自动挑）")
    s.add_argument("repo")
    s.add_argument("--dest", default="")
    s.add_argument("--source", choices=["auto", "hf", "modelscope"], default="auto")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_pull)

    s = sub.add_parser("predict", help="用成本模型算一次 decode")
    s.add_argument("--model", choices=list(MODELS), default="olmoe-1b-7b")
    s.add_argument("--ctx", type=int, default=4096)
    s.add_argument("--batch", type=int, default=1)
    s.add_argument("--w-bits", type=float, default=4.5)
    s.add_argument("--kv-bits", type=float, default=16)
    s.add_argument("--bw", type=float, default=320)
    s.add_argument("--t0", type=float, default=1.5)
    s.add_argument("--tflops", type=float, default=60)
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("fit", help="从实测点反解 t0 与有效带宽")
    s.add_argument("--points", required=True, help="JSON: [{label,bytes,ms}]")
    s.add_argument("--nominal-bw", type=float, default=None,
                   help="不给就从 runs/env.json 的卡名自动匹配")
    s.add_argument("--env", default="runs/env.json")
    s.add_argument("--out", default="")
    s.set_defaults(func=cmd_fit)

    s = sub.add_parser("sweep", help="vLLM 吞吐扫描")
    s.add_argument("--model", required=True)
    s.add_argument("--batches", default="1,4,8,16")
    s.add_argument("--max-tokens", type=int, default=256)
    s.add_argument("--kv-dtype", default="auto")
    s.add_argument("--dtype", default="auto")
    s.add_argument("--max-len", type=int, default=4096)
    s.add_argument("--ctx-tokens", type=int, default=0,
                   help="把 prompt 填到约这么多 token，用来量长上下文下的 decode 成本（0=用默认短 prompt）")
    s.add_argument("--gpu-util", type=float, default=0.90)
    s.add_argument("--tp", type=int, default=1)
    s.add_argument("--eager", action="store_true")
    s.add_argument("--prefix-caching", action="store_true",
                   help="默认关闭：测吞吐时别让前缀缓存影响可复现性")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--out", default="")
    s.set_defaults(func=cmd_sweep)

    s = sub.add_parser("llama", help="llama-bench 单点")
    s.add_argument("--bin-dir", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--n-prompt", type=int, default=512)
    s.add_argument("--n-gen", type=int, default=256)
    s.add_argument("--reps", type=int, default=5)
    s.add_argument("--ngl", type=int, default=99)
    s.add_argument("--n-cpu-moe", type=int, default=None)
    s.add_argument("--threads", type=int, default=None)
    s.add_argument("--ctx", type=int, default=4096)
    s.add_argument("--flash-attn", default="", choices=["", "on", "off", "auto"])
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--out", default="")
    s.set_defaults(func=cmd_llama)

    s = sub.add_parser("offload", help="MoE 专家卸载曲线")
    s.add_argument("--bin-dir", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--points", default="0,4,8,12,16")
    s.add_argument("--n-gen", type=int, default=128)
    s.add_argument("--reps", type=int, default=3)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--out", default="")
    s.set_defaults(func=cmd_offload)

    s = sub.add_parser("ppl", help="perplexity")
    s.add_argument("--bin-dir", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--corpus", required=True)
    s.add_argument("--ctx", type=int, default=512)
    s.add_argument("--ngl", type=int, default=99)
    s.add_argument("--out", default="")
    s.set_defaults(func=cmd_ppl)

    s = sub.add_parser("tasks", help="任务集通过率")
    s.add_argument("--engine", choices=["llamacpp", "vllm"], default="llamacpp")
    s.add_argument("--bin-dir", default="")
    s.add_argument("--model", required=True)
    s.add_argument("--tasks", required=True)
    s.add_argument("--max-tokens", type=int, default=192)
    s.add_argument("--max-len", type=int, default=4096)
    s.add_argument("--ngl", type=int, default=99)
    s.add_argument("--out", default="")
    s.set_defaults(func=cmd_tasks)

    s = sub.add_parser("report", help="汇总 runs 目录")
    s.add_argument("dir", nargs="?", default="runs")
    s.set_defaults(func=cmd_report)
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
