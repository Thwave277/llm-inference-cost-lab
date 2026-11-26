# LLM Inference Cost Lab

Measured, reproducible cost accounting for LLM inference — no answer caching, no template reuse, every token generated.

The question this repo answers is not "how many times can caching save you", but: **given a fixed GPU, where is the floor for each optimization, and which layers are already at that floor.**

Every number here comes from a real run on hardware I own or rented: an RTX 5060 Laptop (8 GB) and a rented RTX 4090 (24 GB).
The report ships with the raw `runs/` JSON, the probes that produced them, and a one-command self-check.

**[Full report (Chinese)](costmodel/REPORT.md)** · **[Two-page brief](costmodel/BRIEF.md)** · **[How to reproduce](REPRODUCE.md)**

---

## Findings

| Claim | Measured | Where |
|---|---|---|
| Decode is bandwidth-bound; every step re-reads the weights | 801 GB/s on 4090 (79% of the 1008 spec), 320–343 GB/s on the laptop | 6.2 / 6.3 |
| Quantization is near-linear, not "gentle" | Q4→Q8: bytes ×1.65, time ×1.45 (the delta is fixed overhead) | 5.x |
| Batching is the cheapest lever that exists | batch 1→32 = 26.7× throughput (dense), 18.3× (MoE), step time +20% | 6.2 / 6.7 |
| The real ceiling is KV capacity, not compute | 50,160 tokens on 24 GB; batch 8 peaks at 8k ctx, batch 16 collapses | 6.8 |
| fp8 KV is a net loss on Ada + vLLM | −21~26%: enabling it drops FlashAttention for XFormers | 6.3 |
| …but the same trick wins in llama.cpp | q8 KV costs only 8.9% and halves the cache | 6.10 |
| MoE expert offloading is a cliff, not a slope | offloading layer 1 already costs −83%; all-CPU keeps 3% | 6.4 |
| Power is a throughput problem | 283 → 301 W (+6%) while throughput goes 26×, so 24.6× less energy per token | 6.11 |
| Speculative decoding is not free | 1.05× on free-form generation, 1.5–1.9× on copy-heavy text | 6.12 |
| INT4 hurts confidence, not correctness | PPL +12.6% on a 4B, but only 1 of 32 multiple-choice items flips | 6.13 |
| Attention becomes the bottleneck at | ≈14.8k (1.7B) / ≈21.7k (4B) tokens for prefill | 7.4 |

**The short version:** on the current Transformer + KV-cache paradigm, batching and quantization are already squeezed dry.
What remains is a capacity problem (KV) and a structural one (prefill compute / architecture), not a clever-kernel problem.

---

## What's inside

```
costmodel/     zero-dependency analytic model (Node) + report + docx exports
edgebench/     measurement toolkit: env, doctor, sweep, offload, ppl, tasks, probes
edgebench/runs-4090/  raw results — 29 files, byte-for-byte as produced
REPRODUCE.md   three tiers: no GPU → 8 GB laptop → rented 24 GB
```

## Quickstart

No GPU needed:

```bash
cd costmodel
node costmodel.mjs --selftest                       # 6/6 passed
node costmodel.mjs --model qwen3-1.7b --ctx 4096    # predicted ms/token, tok/s, bottleneck
```

Verify the package (scripts compile, JSON parses, report references resolve):

```bash
python edgebench/selftest.py
```

On a GPU, see [REPRODUCE.md](REPRODUCE.md).

## Environment

All numbers are locked to: vLLM 0.10.2 · transformers 4.55.4 · torch 2.8.0+cu128 · Python 3.10.8 · CUDA 11.8 · Ollama 0.40.16 · Node 24.
See `edgebench/requirements.txt`. **Cross-version comparisons must re-run `calibrate.mjs`** — these figures are tied to that stack.

## License

Code: MIT (see [LICENSE](LICENSE)). Report text and figures: CC BY 4.0 — reuse with attribution.
