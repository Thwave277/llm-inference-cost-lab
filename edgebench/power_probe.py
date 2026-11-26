"""功耗采样：同一批请求下测 tok/s 和 J/token"""
import argparse
import json
import statistics
import subprocess
import threading
import time

from vllm import LLM, SamplingParams

PROMPT = "用三句话解释显存带宽为什么是解码阶段的瓶颈。"


def sample_power(stop, samples, i):
    cmd = ["nvidia-smi", "--query-gpu=power.draw,clocks.sm,utilization.gpu",
           "--format=csv,noheader,nounits", "-i", str(i)]
    while not stop.is_set():
        r = subprocess.run(cmd, capture_output=True, text=True)
        line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        try:
            p, c, u = [float(x.strip()) for x in line.split(",")]
            samples.append((time.time(), p, c, u))
        except Exception:
            pass
        time.sleep(0.1)


def window_mean(samples, t0, t1, idx=1):
    vals = [s[idx] for s in samples if t0 <= s[0] <= t1]
    return statistics.fmean(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batches", default="1,4,8,16,32")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    samples = []
    stop = threading.Event()
    thr = threading.Thread(target=sample_power, args=(stop, samples, a.gpu), daemon=True)
    thr.start()
    time.sleep(2)
    idle_w = window_mean(samples, 0, time.time() + 1)
    print(f"空载功耗 {idle_w:.1f} W", flush=True)

    llm = LLM(model=a.model, max_model_len=2048, gpu_memory_utilization=0.9,
              max_num_seqs=64, disable_log_stats=True)
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=0.0)
    rows = []
    for b in [int(x) for x in a.batches.split(",") if x.strip()]:
        prompts = [PROMPT] * b
        llm.generate(prompts, sp)  # warmup
        ts, ws = [], []
        for _ in range(a.reps):
            t0 = time.time()
            out = llm.generate(prompts, sp)
            t1 = time.time()
            n = sum(len(o.outputs[0].token_ids) for o in out)
            ts.append((t1 - t0) * 1000.0 / n)   # ms per token (batch step / b)
            ws.append(window_mean(samples, t0, t1, 1))
        step_ms = statistics.median(ts) * b
        ms_tok = statistics.median(ts)
        w = statistics.median(ws)
        j_tok = w * (ms_tok / 1000.0)
        rows.append({"batch": b, "step_ms": round(step_ms, 2), "ms_token": round(ms_tok, 4),
                     "tok_s": round(1000.0 / ms_tok, 1), "watt": round(w, 1),
                     "j_token": round(j_tok, 4), "idle_watt": round(idle_w, 1)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    stop.set()
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("=== 汇总")
    base = rows[0]["j_token"]
    for r in rows:
        print(f"batch {r['batch']:>3}  {r['tok_s']:>7.1f} t/s  {r['watt']:>6.1f} W  "
              f"{r['j_token']*1000:>7.2f} mJ/token  (相对 b=1: {base/r['j_token']:.2f}×)")


if __name__ == "__main__":
    main()
