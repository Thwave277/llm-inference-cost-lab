# LLM 本地推理成本实验室

**[English](README.md)**

一句话：在"每次请求都真正自主生成、不缓存答案、不复用模板"的前提下，把大模型推理的算力成本拆到可计算的粒度，
再用消费级显卡实测校验。结论不是"能省多少倍"，而是"每一层的地板在哪、哪一层已经到底"。

主报告：[costmodel/REPORT.md](costmodel/REPORT.md)（导出件 `推理算力成本评估报告.docx`，第 0.1 节是选型决策矩阵）
精简版：[costmodel/BRIEF.md](costmodel/BRIEF.md)（导出件 `推理算力成本评估报告-精简版.docx`，两页）
复现指引：[REPRODUCE.md](REPRODUCE.md)（按硬件分三档，每档给命令 + 预期数字 + 耗时花费）
数据索引：[edgebench/RUNS.md](edgebench/RUNS.md)（每个 json 是哪条命令跑的、对应哪一节）
整包分发件：`llm-cost-lab-v1.9.zip`（含报告、脚本、全部实测数据，CRC 校验通过）

---

## 结论速查

| 结论 | 数字 | 出处 |
|---|---|---|
| decode 是带宽受限，每步要把权重读一遍 | 4090 实测 801 GB/s（标称 79%）；5060 实测 320–343 GB/s | 6.2 / 6.3 |
| 量化接近线性，不是"温和降功耗" | Q4→Q8 字节 ×1.65，时间 ×1.45（差值就是固定开销） | 5.x / 6 |
| 每步固定开销 | 1.5–2.1 ms（小模型上占 28%） | 5.2 |
| dense 的 batch 杠杆 | batch 1→32 吞吐 26.7×，单步只涨 20% | 6.2 |
| MoE 的 batch 杠杆 | 18.3×（热门专家让并集只覆盖 42.9%，不是均匀路由的 87.3%；hook 直测） | 6.7 |
| 真正的天花板是 KV 容量 | 30B-A3B 在 24 GB 卡上实测 50,160 token；8k 上下文时 batch 8 是峰值，batch 16 崩塌 | 6.8 |
| fp8 KV 在 vLLM+Ada 上是净亏 | −21~26%（丢 FlashAttention，退到 XFormers） | 6.3 |
| 同一件事在 llama.cpp 里是赚的 | q8_0 KV 只慢 8.9%，KV 减半 | 6.10 |
| MoE 专家卸载是断崖 | 卸载第 1 层就 −83%，全 CPU 剩 3% | 6.4 |
| 功耗几乎不随 batch 变 | 283 → 301 W（+6%），吞吐 26×，能效 24.6× | 6.11 |
| 电费不是成本 | 一次 128 token 约 0.00007 元，卡时是电费的三千倍 | 6.11 |
| 投机解码不是白拿 | 自由生成 1.05×，复述型 1.5–1.9×（取决于文本可预测性） | 6.12 |
| prefill 是另一种病：算力受限 | 5060 上 dense 34–36 TFLOPS（标称 57–60%），MoE 22.5（37%）；固定开销 8–15 ms | 7.2 |
| 长 prompt 何时开始贵 | 注意力追平线性项：1.7B ≈ 1.48 万 token、4B ≈ 2.17 万 | 7.4 |
| 量化伤在哪 | W4A16 在 4B 上 PPL +12.6%，但多选题只差 1/32——损伤在信心不在对错 | 6.13 |
| 精度准入不能用任务准确率 | 只看 argmax 测不出分布退化，必须带 PPL 或 KL | 6.13 |

一句话总结：**在现有 Transformer 上，能拿的都拿了——剩下的是容量和结构问题，不是算法问题。**

要「我现在该选什么」的答案，直接看 [REPORT.md 第 0.1 节](costmodel/REPORT.md) 的选型决策矩阵（6 个场景）。

---

## 目录

```
llm-cost-lab/
├── costmodel/                 # 成本模型：算，不跑模型
│   ├── costmodel.mjs          # 零依赖 CLI（predict / 预设 / 自测）
│   ├── costmodel.cmd          # Windows 包装
│   ├── calibrate.mjs          # 连 Ollama 反解真实带宽
│   ├── longctx.mjs            # 量 KV 的边际成本
│   ├── md2docx.py             # REPORT.md → docx（单源维护）
│   ├── BRIEF.md               # 两页精简版
│   ├── REPORT.md              # 主报告（v1.9）
│   └── README.md              # 模型本身的说明：公式、参数、局限
├── edgebench/                 # 实测工具：跑模型，出数字
│   ├── edgebench/             # 包：env/doctor/pull/predict/fit/sweep/llama/offload/ppl/tasks/report
│   ├── configs/               # quick.json（40 分钟冒烟）、tasks.jsonl（6 题）
│   ├── power_probe.py         # 功耗采样（tok/s → J/token）
│   ├── spec_probe.py          # 投机解码对照（ngram）
│   ├── router_probe.py        # 路由热度直测（hook FusedMoE.select_experts）
│   ├── ppl_probe.py           # perplexity（wikitext-2）
│   ├── mc_probe.py            # 多选题打分（base 模型也能测精度）
│   ├── ppl_check.py           # PPL 交叉校验（transformers vs vLLM）
│   ├── runs-4090/             # 两台机器上的全部实测数据（29 个文件）
│   ├── selftest.py            # 包自检：脚本能编译、数据在、报告引用不落空
│   ├── RUNS.md                # 数据索引：文件 → 命令 → 报告章节
│   ├── requirements.txt       # 冻结的版本（vLLM 0.10.2 / transformers 4.55.4 / torch 2.8.0）
│   └── HANDOFF.md             # 给"另一台机器上的 agent"看的交接指令
├── REPRODUCE.md               # 三档复现指引
└── llm-cost-lab-v1.9.zip      # 整包分发件
```

---

## 怎么用

本地（只有 Windows 机器时，不需要 GPU）：

```bash
cd costmodel
costmodel.cmd --model qwen3-1.7b --ctx 4096          # 预测基线
costmodel.cmd --model deepseek-v3 --wbits 8          # 换模型/精度
costmodel.cmd --selftest                              # 6 条公式自测
```

有 GPU 时（本机或租的机器）：

```bash
cd edgebench
python -m edgebench doctor                            # 环境体检（先跑这个）
python -m edgebench sweep --model <路径> --batches 1,8,32
python -m edgebench llama --bin-dir <llama.cpp/bin> --model <gguf>   # llama-bench 包装
python -m edgebench offload --bin-dir <bin> --model <gguf>           # MoE 卸载曲线
python power_probe.py --model <路径>                  # 功耗
python spec_probe.py --model <路径> --method ngram     # 投机解码
```

远端（AutoDL）的 SSH 配置写成本地 `~/.ssh/config` 的 Host 块，别名 `autodl`，可直接 `ssh autodl`。

---

## 数据口径（重要）

- **两台机器**：本地 RTX 5060 Laptop 8 GB（Windows，Ollama/llama.cpp）、远端 RTX 4090 24 GB（Ubuntu 22.04，vLLM 0.10.2 + 自建 llama.cpp）
- **计时方法**：不信任引擎自报的 tok/s。vLLM 用"两个生成长度做差"消掉 prefill；llama.cpp 用 `llama-bench`
- **噪声 20–40%**：笔记本功耗墙、云端共享卡。所有数字取中位数，小数点后第二位不可信
- **不可横比**：vLLM 与 llama.cpp 的口径（量化格式、调度器、反量化路径）不同，只看趋势不看绝对值
- **成本模型与实测的偏差**：predict 在 dense 上 4% 内，在 MoE 上会因为路由偏斜低估 batch 收益

---

## 已知缺口

| 缺口 | 为什么没做 |
|---|---|
| 量化精度（PPL / 任务集） | 需要 fp16 对照权重，24 GB 装不下 30B；4B 的对照代表性不足 |
| 32k 以上的 prefill | 只测到 4k，交叉点之后的曲线是外推 |
| 稀疏／动态注意力 | vLLM/Ollama 不直接暴露开关，要 SGLang 或改 backend |
| EAGLE／MTP 形态的投机解码 | vLLM 0.10 移除了草稿模型方式，这些要另配权重 |
| MLA 的实际 KV 容量 | DeepSeek-V2-Lite 在 ModelScope 只有 bf16（31 GB），盘不够 |
| 多卡／EP | 只有单卡数据，all-to-all 成本未覆盖 |

---

## 怎么验证这份东西没写错

```bash
cd costmodel && costmodel.cmd --selftest      # 6/6 通过说明公式没退化
cd edgebench && python -m edgebench env       # 看本机硬件是否被正确识别
```

报告里每个数字都能在 `edgebench/runs-4090/` 找到原始 jsonl；章节号对应 REPORT.md。
