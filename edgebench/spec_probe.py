"""投机解码对照：加/不加 spec（draft model 或 ngram）的 tok/s"""
import argparse
import json
import statistics
import time

from vllm import LLM, SamplingParams

GENERIC = "用三句话解释显存带宽为什么是解码阶段的瓶颈。"

# 抄袭型任务：要求复述给定的段落，ngram 投机最容易命中的场景
PARA = (
    "本地推理的成本由三件事决定：权重有多少字节必须从显存读出来，"
    "上下文增长时 KV 缓存要多读多少，以及每一步有多少固定开销摊在 batch 上。"
    "带宽决定了下限，容量决定了上限，调度决定了你能不能同时吃到两者。"
)
COPY = f"请原样重复下面这段话，不要改动任何字：\n{PARA}\n\n原文："


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", default="", help="ngram 或空")
    ap.add_argument("--draft", default="")
    ap.add_argument("--num-spec", type=int, default=5)
    ap.add_argument("--prompt", default="generic", choices=["generic", "copy"])
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    kw = dict(model=a.model, max_model_len=2048, gpu_memory_utilization=0.9,
              max_num_seqs=64, disable_log_stats=True)
    if a.draft:
        kw["speculative_config"] = {"model": a.draft, "num_speculative_tokens": a.num_spec}
    elif a.method:
        kw["speculative_config"] = {"method": a.method, "num_speculative_tokens": a.num_spec}

    t0 = time.time()
    llm = LLM(**kw)
    load = time.time() - t0
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=0.0, ignore_eos=True)
    text = GENERIC if a.prompt == "generic" else COPY

    rows = []
    for b in [int(x) for x in a.batches.split(",") if x.strip()]:
        prompts = [text] * b
        llm.generate(prompts, sp)
        ts = []
        for _ in range(a.reps):
            t = time.time()
            out = llm.generate(prompts, sp)
            n = sum(len(o.outputs[0].token_ids) for o in out)
            ts.append((time.time() - t) * 1000.0 / n)
        ms = statistics.median(ts)
        rows.append({"batch": b, "prompt": a.prompt, "method": a.method or ("draft" if a.draft else "none"),
                     "ms_token": round(ms, 4), "tok_s": round(1000.0 / ms, 1), "load_s": round(load, 1)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
