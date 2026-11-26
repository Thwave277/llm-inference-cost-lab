#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""精度探针：perplexity（wikitext-2 test）+ 一组确定性任务，同一引擎下比较 FP16 与量化。

用法:
  python ppl_probe.py --model <path> --corpus /root/autodl-tmp/wikitext2-test.txt \
      --chunks 32 --chunk-tokens 1024 --max-len 2048 --out runs/ppl-4b-bf16.json
"""
import argparse
import json
import math
import os
import re
import time

import torch
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--chunks", type=int, default=32)
ap.add_argument("--chunk-tokens", type=int, default=1024)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--gpu-util", type=float, default=0.90)
ap.add_argument("--skip-tasks", action="store_true")
ap.add_argument("--out", required=True)
args = ap.parse_args()

# ---------- 任务集：确定性、可精确判分 ----------
def build_tasks():
    T = []
    for a, b in [(17, 23), (48, 37), (96, 12), (123, 45), (7, 89),
                 (256, 4), (63, 64), (1000, 3), (19, 19), (72, 25),
                 (144, 12), (55, 55), (81, 27), (108, 9), (39, 41)]:
        T.append({"kind": "math", "q": "Answer with the number only: %d * %d =" % (a, b), "expect": str(a * b)})
    def fib(n):
        a, b = 1, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return a

    for n in [8, 13, 21]:
        T.append({"kind": "math", "q": "What is the %d-th Fibonacci number (1,1,2,3,...)? Answer with the number only." % n,
                  "expect": str(fib(n))})

    code_cases = [
        ("x = [1, 2, 3]\nprint(sum(x))", "6"),
        ("print(len('hello'))", "5"),
        ("print(2 ** 10)", "1024"),
        ("a = [3, 1, 2]\nprint(sorted(a)[0])", "1"),
        ("print(type(1) == type(1.0))", "false"),
        ("print('ab' * 2)", "abab"),
    ]
    for code, ans in code_cases:
        T.append({"kind": "code", "q": "What does this Python print? Answer with the output only.\n%s" % code, "expect": ans})

    zh_cases = [
        ("中国最长的河流是哪条？只回答河名。", "长江"),
        ("水的化学式是什么？只回答化学式。", "H2O"),
        ("一年有多少个月？只回答数字。", "12"),
        ("三角形内角和是多少度？只回答数字。", "180"),
        ("光速每秒约多少公里？只回答数字。", "30"),
        ("给出三种颜色的中文名，用逗号分隔。", "红"),
    ]
    for q, a in zh_cases:
        T.append({"kind": "zh", "q": q, "expect": a})

    logic = [
        ("If all cats are animals and Tom is a cat, is Tom an animal? Answer yes or no.", "yes"),
        ("Is 91 a prime number? Answer yes or no.", "no"),
        ("Which is larger: 3/7 or 4/9? Answer with the fraction only.", "4/9"),
        ("If today is Tuesday, what day is it 10 days from now? Answer with the day name only.", "friday"),
    ]
    for q, a in logic:
        T.append({"kind": "logic", "q": q, "expect": a})
    return T


def norm(s):
    s = s.strip().lower()
    s = re.sub(r"[\s,]+", " ", s)
    s = re.sub(r"[。！？,.!?]", "", s)
    return s


def main():
    print("loading model ...", flush=True)
    t0 = time.time()
    llm = LLM(model=args.model, max_model_len=args.max_len, gpu_memory_utilization=args.gpu_util)
    load_s = time.time() - t0
    tok = llm.get_tokenizer()

    # ---------- perplexity ----------
    text = open(args.corpus, encoding="utf-8", errors="ignore").read()
    ids_all = tok.encode(text)[: args.chunks * args.chunk_tokens + 1]
    chunks = []
    for i in range(args.chunks):
        c = ids_all[i * args.chunk_tokens:(i + 1) * args.chunk_tokens]
        if len(c) >= args.chunk_tokens // 2:
            chunks.append(c)

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    nll, ntok, per_chunk = 0.0, 0, []
    for c in chunks:
        out = llm.generate([{"prompt_token_ids": c}], sp, use_tqdm=False)[0]
        plp = out.prompt_logprobs or []
        s, n = 0.0, 0
        for i in range(1, len(c)):
            ent = plp[i] if i < len(plp) else None
            if not ent:
                continue
            lp = ent.get(c[i])
            if lp is None:
                continue
            s += lp.logprob
            n += 1
        nll += -s
        ntok += n
        per_chunk.append(round(math.exp(-s / n), 4) if n else None)
    ppl = math.exp(nll / ntok) if ntok else float("nan")

    result = {
        "model": args.model,
        "load_s": round(load_s, 1),
        "ppl": round(ppl, 4),
        "ppl_tokens": ntok,
        "ppl_chunks": len(chunks),
        "ppl_per_chunk": per_chunk[:8],
    }
    print(json.dumps({k: result[k] for k in ("ppl", "ppl_tokens", "ppl_chunks")}), flush=True)

    # ---------- 任务准确率 ----------
    if not args.skip_tasks:
        tasks = build_tasks()
        sp2 = SamplingParams(temperature=0.0, max_tokens=64, stop=["\n\n"])
        prompts = [t["q"] for t in tasks]
        t1 = time.time()
        outs = llm.generate(prompts, sp2, use_tqdm=False)
        task_s = time.time() - t1
        ok = 0
        fails = []
        for t, o in zip(tasks, outs):
            got = norm(o.outputs[0].text)
            exp = norm(t["expect"])
            hit = exp in got
            ok += hit
            if not hit:
                fails.append({"kind": t["kind"], "q": t["q"][:60], "expect": t["expect"], "got": o.outputs[0].text.strip()[:80]})
        result.update({
            "tasks_total": len(tasks),
            "tasks_ok": ok,
            "task_acc": round(ok / len(tasks), 4),
            "task_s": round(task_s, 1),
            "task_fails": fails[:12],
        })
        print(json.dumps({"task_acc": result["task_acc"], "ok": ok, "n": len(tasks)}), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
