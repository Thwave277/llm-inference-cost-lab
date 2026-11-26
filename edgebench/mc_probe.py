#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多选题打分：base 模型也能测——对每题 4 个选项算平均 logprob，取最高者为答案。
用法: python mc_probe.py --model <path> --out runs/mc-4b-bf16.json
"""
import argparse
import json
import math
import os

from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--gpu-util", type=float, default=0.90)
ap.add_argument("--out", required=True)
args = ap.parse_args()


def items():
    I = []
    # --- 算术：正确值 + 三个近似错项 ---
    for a, b in [(17, 23), (48, 37), (96, 12), (123, 45), (7, 89), (256, 4),
                 (63, 64), (19, 19), (72, 25), (144, 12), (55, 55), (39, 41)]:
        v = a * b
        opts = [v, v + a, v - b, v + 10]
        I.append({"kind": "math", "q": "What is %d × %d?" % (a, b),
                  "opts": ["%d" % o for o in opts], "gold": 0})
    # --- 中文常识 ---
    zh = [
        ("中国最长的河流是？", ["长江", "黄河", "珠江", "黑龙江"], 0),
        ("水的化学式是？", ["H2O", "CO2", "O2", "NaCl"], 0),
        ("一年有多少个月？", ["12", "10", "24", "6"], 0),
        ("三角形内角和是多少度？", ["180", "90", "360", "270"], 0),
        ("光在真空中的速度约为每秒多少公里？", ["30万", "3万", "300万", "3000"], 0),
        ("地球上最大的洋是？", ["太平洋", "大西洋", "印度洋", "北冰洋"], 0),
        ("《红楼梦》的作者是？", ["曹雪芹", "施耐庵", "罗贯中", "吴承恩"], 0),
        ("中国的首都是？", ["北京", "上海", "广州", "南京"], 0),
    ]
    for q, opts, g in zh:
        I.append({"kind": "zh", "q": q, "opts": opts, "gold": g})
    # --- 逻辑 ---
    lg = [
        ("If all cats are animals and Tom is a cat, Tom is", ["an animal", "a plant", "a mineral", "an idea"], 0),
        ("Is 91 a prime number?", ["No", "Yes", "Only when divided by 7", "It is not defined"], 0),
        ("Which is larger?", ["4/9", "3/7", "They are equal", "Cannot be compared"], 0),
        ("If today is Tuesday, ten days later it is", ["Friday", "Monday", "Thursday", "Wednesday"], 0),
        ("All roses are flowers. Some flowers fade quickly. Therefore some roses", ["may fade quickly", "must fade quickly", "never fade", "are not flowers"], 0),
        ("5 machines make 5 widgets in 5 minutes. 100 machines make 100 widgets in", ["5 minutes", "100 minutes", "20 minutes", "1 minute"], 0),
    ]
    for q, opts, g in lg:
        I.append({"kind": "logic", "q": q, "opts": opts, "gold": g})
    # --- 代码 ---
    cd = [
        ("What does this print?\nx = [1, 2, 3]\nprint(sum(x))", ["6", "3", "[1, 2, 3]", "123"], 0),
        ("What does this print?\nprint(len('hello'))", ["5", "4", "6", "hello"], 0),
        ("What does this print?\nprint(2 ** 10)", ["1024", "20", "100", "512"], 0),
        ("What does this print?\na = [3, 1, 2]\nprint(sorted(a)[0])", ["1", "3", "2", "[1, 2, 3]"], 0),
        ("What does this print?\nprint('ab' * 2)", ["abab", "ab", "ab ab", "aabb"], 0),
        ("What does this print?\nprint(int('7') + 1)", ["8", "71", "7", "Error"], 0),
    ]
    for q, opts, g in cd:
        I.append({"kind": "code", "q": q, "opts": opts, "gold": g})
    return I


def main():
    llm = LLM(model=args.model, max_model_len=args.max_len, gpu_memory_utilization=args.gpu_util)
    tok = llm.get_tokenizer()
    data = items()

    prompts, meta = [], []
    for idx, it in enumerate(data):
        for oi, opt in enumerate(it["opts"]):
            prefix = it["q"] + "\nAnswer: "
            text = prefix + opt
            pid = tok.encode(prefix)
            fid = tok.encode(text)
            start = len(pid)
            if fid[:start] != pid:          # BPE 边界对不上时退一格
                start = max(0, start - 1)
            prompts.append({"prompt_token_ids": fid})
            meta.append((idx, oi, start))

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate(prompts, sp, use_tqdm=False)

    scores = {}
    for (idx, oi, start), o in zip(meta, outs):
        plp = o.prompt_logprobs or []
        s, n = 0.0, 0
        for i in range(start, len(plp)):
            ent = plp[i]
            if not ent:
                continue
            # 该位置的真实 token
            tid = o.prompt_token_ids[i]
            lp = ent.get(tid)
            if lp is None:
                continue
            s += lp.logprob
            n += 1
        avg = s / n if n else -99.0
        scores[(idx, oi)] = avg

    ok, rows = 0, []
    for idx, it in enumerate(data):
        vals = [scores[(idx, oi)] for oi in range(len(it["opts"]))]
        pick = max(range(len(vals)), key=lambda i: vals[i])
        hit = pick == it["gold"]
        ok += hit
        rows.append({"kind": it["kind"], "q": it["q"][:50], "pick": it["opts"][pick],
                     "gold": it["opts"][it["gold"]], "ok": bool(hit),
                     "margin": round(vals[pick] - sorted(vals)[-2], 4)})

    res = {
        "model": args.model,
        "n": len(data),
        "ok": ok,
        "acc": round(ok / len(data), 4),
        "by_kind": {},
        "rows": rows,
    }
    for k in ("math", "zh", "logic", "code"):
        sub = [r for r in rows if r["kind"] == k]
        res["by_kind"][k] = round(sum(r["ok"] for r in sub) / len(sub), 4) if sub else None
    res["mean_margin"] = round(sum(r["margin"] for r in rows) / len(rows), 4)
    print(json.dumps({k: res[k] for k in ("n", "ok", "acc", "by_kind", "mean_margin")}, ensure_ascii=False), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
