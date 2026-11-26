# 数据索引：每个 json 是哪条命令跑出来的、对应报告哪一节

所有文件都在 `edgebench/runs-4090/`（原样拉回，未改一个字节），
另有 `edgebench/runs/` 是笔记本侧产出的环境快照与带宽拟合。

| 文件 | 生成命令 | 内容 | 报告 |
|---|---|---|---|
| `sweep-2026112*.jsonl`（7 个） | `python -m edgebench sweep --model <4B 路径> --batches 1,4,8,16,32` | dense 4B 的 batch 曲线（含长上下文、fp8 KV 对照组） | 6.2 / 6.3 |
| `moe-batch-vllm-gptq.jsonl` | `python -m edgebench sweep --model <30B-A3B GPTQ-Int4> --batches 1,4,8,16,32` | MoE 的 batch 曲线，18.3× | 6.7 |
| `moe-ctx8192-vllm.jsonl` | 同上 + `--ctx-tokens 8192` | 容量墙：batch 8 峰值、batch 16 崩塌 | 6.8 |
| `bench-offload-30ba3b.json` | `python -m edgebench offload --bin-dir <build/bin> --model <gguf>` | `--n-cpu-moe` 全扫描（0…48） | 6.4 |
| `bench-offload-cliff.json` | `llama-bench -ngl 99 --n-cpu-moe 0/1/2` | 断崖前三点，确认不是斜坡 | 6.4 |
| `power-4b.jsonl` | `python power_probe.py --model <4B 路径> --batches 1,4,8,16,32` | 每 token 能耗 3,077 → 125 mJ | 6.11 |
| `spec-base.jsonl` / `spec-ngram-gen.jsonl` | `python spec_probe.py --method ngram`（自由生成） | 投机解码自由生成 1.05× | 6.12 |
| `spec-base-copy.jsonl` / `spec-ngram-copy.jsonl` | 同上（复述型任务） | 复述型 1.5–1.9× | 6.12 |
| `ppl-4b-bf16.json` / `ppl-4b-awq.json` | `python ppl_probe.py --model … --corpus wikitext2-test.txt` | PPL 15.17 → 17.08 | 6.13 |
| `mc-4b-bf16.json` / `mc-4b-awq.json` | `python mc_probe.py --model …` | 多选题 28/32 → 27/32 | 6.13 |
| `router-gptq.json` | `python router_probe.py --model <30B-A3B GPTQ-Int4>` | 专家并集 6.25% / 34.4% / 42.9% | 6.7 |
| `prefill-qwen3-1.7b.json` / `-full` / `-long` | `node ../costmodel/prefill.mjs --model qwen3-1.7b` | 笔记本 prefill 曲线 | 7.2 |
| `prefill-qwen3-4b.json` / `-full` / `-long` | 同上，换 `--model qwen3-4b` | 同上 | 7.2 |
| `prefill-olmoe-1b-7b.json` / `-full` | 同上，换 `--model olmoe` | MoE 的 prefill 只有 dense 的 2/3 效率 | 7.2 |
| `runs/env.json` | `python -m edgebench env` | 本机 GPU / 驱动 / 版本快照 | 6.1 |
| `runs/fit.json` | `python -m edgebench fit` | 三个真实点拟合出 t0=0.83 ms、BW=256 GB/s | 5.2 |

## 口径提醒

- `sweep-*.jsonl` 是**追加写**的：同一个文件里可能有多组实验，读的时候按 `tag` 字段筛
- 笔记本数据（`prefill-*`）是 Ollama 的 `prompt_eval_duration` 折算的，**不可**与 4090 的 vLLM 数字横比
- `router-gptq.json` 那轮跑的是 `enforce_eager=True`，其中的 `tok_s` 字段不可用于性能结论，只有并集统计有效
- 所有文件的生成时间在文件名或内部 `stamp` 字段里，对不上时间的就是两轮不同环境
