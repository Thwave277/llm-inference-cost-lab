# edgebench

消费级 / 租用显卡上的 LLM 推理基准与成本校准。**只测量，不造引擎。**

核心 CLI 零第三方依赖（torch / vllm 都是延迟导入，没装的机器也能 `--help`）；但真正跑测量要先装 vLLM 和 torch——`doctor` 会把缺什么、装哪个版本告诉你。

## 装

```bash
python -m edgebench env --out runs/env.json   # 先留证：卡、驱动、PCIe、版本
bash setup.sh                                  # 装 vllm；要 llama.cpp 加 --with-llamacpp
```

## 常用命令

| 命令 | 干什么 |
|---|---|
| `doctor` | 兼容性体检：卡 sm / torch 编译目标 / vLLM 版本 / 二进制 / 磁盘。**换机器先跑它** |
| `env` | 环境快照 + 锁频提示。**所有性能数字必须配它** |
| `pull` | 下模型，自动挑 modelscope → hf-mirror |
| `predict` | 用成本模型算一次 decode 的理论值 |
| `fit` | 从实测点反解固定开销 `t0` 和有效带宽 `BW` |
| `sweep` | vLLM 吞吐扫描，输出 jsonl |
| `llama` | llama-bench 单点 |
| `offload` | MoE 专家卸载曲线（`--n-cpu-moe` 扫描） |
| `ppl` | perplexity |
| `tasks` | 小任务集通过率 |
| `report` | 汇总 `runs/` 里的 jsonl |

## 计时口径（别改）

不信任引擎自报的 tok/s。用**两个生成长度做差**：

```
step_ms = (wall(max_tokens=256) - wall(max_tokens=8)) / (256 - 8)
ms_per_token = step_ms / batch
```

好处：模型加载、prefill、引擎启动开销全被扣掉，剩下的才是纯 decode 每步耗时。

## 成本模型

与 `D:\deepseek\llm-cost-lab\costmodel\costmodel.mjs` 同一套公式：

```
bytes = 激活参数 × bits/8 + batch × (ctx + k) × 每 token KV 字节
flops = 2 × 激活参数 × P + 4 × 层数 × P × ctx × hidden      # P = batch × (1+k)
t_step = max(bytes/BW, flops/TFLOPS) + t0
```

`BW` 用**实测反解值**，不用标称。本机（5060 Laptop）实测 256–343 GB/s，标称 448。

## 判据

- 提速有没有效：同 `batch`、同 `ctx`、同权重，只改一个变量，看 `step_ms`。
- 提速能不能用：PPL 涨 ≤2%、任务集掉 ≤1 题（`precision.judge`）。
- 数字可不可信：`runs/env.json` 在、跑够 3 次、离散度写出来。

## 一页流程

见 `HANDOFF.md`——那是给租用机器上的 DSH 的执行手册，含 6–8 小时的完整实验计划、验收标准和失败排查。
