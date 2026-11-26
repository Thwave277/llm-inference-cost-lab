# 交接指令：在 AutoDL 上把推理性能测出来

> 这份文件是给"租用机器上的 DSH"看的。读完它就是全部上下文，不需要再问上一轮发生了什么。

## 0. 一句话

在一台 24GB（或更大）的租用 GPU 上，用现成推理引擎跑出四组真实数字：**吞吐拐点、KV 量化收益、MoE 专家卸载曲线、精度代价**。不写新引擎。

## 1. 交付物（三样，缺一不算完成）

| # | 交付物 | 形式 |
|---|---|---|
| D1 | `runs/` 目录：所有原始测量结果 | JSON / JSONL，字段见第 6 节 |
| D2 | `REPORT-<日期>.md`：四张表 + 结论 | Markdown，模板见第 7 节 |
| D3 | 一段 200 字以内的口头结论：**这台卡上最值得做的一件事是什么** | 直接写在最终回复里 |

## 2. 已有事实（别重复验证，别推翻）

来源：一台 RTX 5060 Laptop 8GB 的实测，模型 Qwen3-1.7B / 4B / OLMoE-1B-7B，引擎 Ollama。

- 有效带宽 **256–343 GB/s**（标称 448 的 57–77%）。**任何 kernel 优化的天花板由这个数定死。**
- 拟合 `t ≈ 0.8–1.5 ms + bytes / 320GB/s`：每步有 **1–2ms 固定开销**（kernel 启动 / 采样 / 调度），在小模型上占 28%。
- 量化：Q4→Q8 字节 ×1.65、时间 ×1.45，**接近线性**（decode 是带宽瓶颈）。
- batch 越大越省，但存在拐点：`bytes/token ≈ batch × KV` 赶超权重读取之后转算力瓶颈。
- MoE：OLMoE（6.9B 总参 / 1.3B 激活）实测 244 tok/s，成本模型预测 254，**误差 4%** —— 模型可用，但要用**实读权重**（GGUF 文件大小 × 激活比），不能拿总参算。
- KV 量化在 1k 上下文内**没有收益**，3 万 token 后才值。

## 3. 硬约束

1. **不要写引擎**。不要实现 PagedAttention / CUDA Graph / 算子融合。vLLM、SGLang、llama.cpp 里都有，你自己写的版本不会更快。你的任务是**测**，不是**造**。
2. **每个数字必须带 `runs/env.json`**。卡型、显存、驱动、PCIe 代际变了，数字就不可比。
3. **同配置至少 3 次，报中位数 + 离散度**。单次结果直接作废。离散度 >20% 要在报告里标红。
4. **先 smoke 再全量**。第 5 节 T1 不通过，禁止往下跑，先修。
5. **不许改机器全局配置**（驱动、CUDA、系统参数）。锁频失败就退回多次取中位数。
6. **推理时别开 monitor 刷屏**。nvidia-smi 每 100ms 一次会拉低 2–5% 吞吐。

## 4. 环境

AutoDL 镜像一般是 Ubuntu + conda + torch 已装。工作目录用 `/root/autodl-tmp`（数据盘，不占系统盘）。

```bash
# 1) 把 edgebench 传上去（scp / 网页上传 / jupyter 上传都行），然后：
cd /root/autodl-tmp/edgebench
bash setup.sh                  # 装 vllm；要 llama.cpp 再加 --with-llamacpp

# 2) 记录环境
export HF_ENDPOINT=https://hf-mirror.com   # 国内加速，下不动再换 modelscope
python -m edgebench env --out runs/env.json
```

**模型选择规则**（自己挑，但必须记下实际用的 repo id 和文件大小）：

| 用途 | 规模 | 备注 |
|---|---|---|
| batch 曲线 | dense 4B–8B（bf16，8–16GB） | 装得下的最大 dense 模型 |
| KV 量化 | 同上 | 需要长上下文对照 |
| MoE 卸载 | 一个**装不进显存**的 MoE GGUF（比如 30B-A3B 的 Q8 ≈32GB 跑 24GB 卡） | 只有溢出才有卸载曲线 |
| 精度回归 | 与 batch 曲线同一个模型 | 必须同权重，否则不可比 |

## 4.5 兼容性（换机器最容易翻车的地方）

先跑体检，全绿再往下：

```bash
python -m edgebench doctor      # FAIL 必须修，WARN 可以带病跑
```

**算力等级与 torch / vLLM 的对应**（对不上就会出现 "no kernel image is available"）：

| 卡 | sm | 要求 |
|---|---|---|
| V100 / T4 | 7.0 / 7.5 | 能跑，但要 `--dtype float16`（**没有 BF16**）；**没有 FlashAttention、没有 FP8**，KV-fp8 那组实验做不了 |
| A100 / A30 | 8.0 | 常规 CUDA 11.8 / 12.1 轮子都行 |
| 3090 / A10 / A40 | 8.6 | 同上 |
| 4090 / L40S / 6000 Ada | 8.9 | 同上 |
| H100 / H200 | 9.0 | 需要 CUDA 12.x 的 wheel |
| 5090 / B40 | 12.0 | 需要 torch ≥ 2.7 + CUDA ≥ 12.8，老镜像直接用不了 |

**版本坑**：
- Qwen3 系列要 vLLM **≥ 0.8.4**；镜像里若是 0.6.x，换 Qwen2.5-7B，别硬上。
- `kv_cache_dtype="fp8"` 在 ≥0.6 都有；更老的只有 `fp8_e5m2`。
- llama.cpp 的 `--n-cpu-moe` 是 2024 年底才加的，必须用新构建；老构建会直接报 unknown argument。
- `llama-bench -fa` 在部分版本要求带值（`on/off/auto`），所以工具默认不传这个参数——跨版本对比时保持默认。

**Windows 不适用**：vLLM 只支持 Linux。本地 Windows 上只有 `doctor` / `env` / `predict` / `fit` 能跑，真正测量必须在 Linux 机器上。

**PCIe 宽度会改变结论**：本机实测是 **PCIe 4.0 x4**（不是 x16）。专家卸载走的就是这条线，x4 的实测带宽约 6 GB/s，比 x16 差 4 倍。租的卡上先看 `runs/env.json` 里的 `pcie.link.width.current`，如果是 x1/x4，MoE 卸载那组实验的收益会被腰斩——这不是代码问题，是物理。

## 5. 执行顺序（预计 6–8 GPU 小时）

### T0 体检（5 分钟）

```bash
python -m edgebench doctor
python -m edgebench env --out runs/env.json
```

`doctor` 出 FAIL 就别往下走。它同时会告诉你：卡是什么 sm、torch 编没编这个 sm、vLLM 版本认不认你要用的模型、llama.cpp 二进制在不在、磁盘够不够。

下模型用内置命令（自动挑 modelscope → hf-mirror）：

```bash
python -m edgebench pull Qwen/Qwen3-4B --dest /root/autodl-tmp/models/Qwen3-4B
```

### T1 冒烟（20 分钟，不通过就停）
```bash
python -m edgebench predict --model qwen3-4b --bw 320 --t0 1.5
python -m edgebench sweep --model <你的模型> --batches 1,4 --max-tokens 64
```
- 通过标准：能出 `step_ms` 和 `tok_s`；batch 4 的 tok/s ≥ batch 1 的 1.5 倍。达不到说明引擎没在 batching，先查 `--gpu-util` / `--max-len` / 模型是否 `enforce_eager`。
- 例外：如果 `step_ms` 也涨了 3 倍以上，那是显存/KV 容量不够，不是 batching 失败。把 `--max-len 2048` 重跑，涨幅恢复就记为 WARN 而不是 FAIL（详见 `configs/quick.json` 的 caveat）。

### T2 吞吐拐点（1.5 小时）
```bash
python -m edgebench sweep --model <模型> --batches 1,2,4,8,16,32 --max-tokens 256 --out runs/sweep-b1.jsonl
python -m edgebench sweep --model <模型> --batches 1 --max-len 8192 --max-tokens 256 --out runs/sweep-ctx8k.jsonl
```
要回答：**拐点在哪**。tok/s 随 batch 增长什么时候不再涨、开始只涨延迟。

前缀缓存默认关闭（`--prefix-caching` 可以打开）。理由不是"会算错"——做差法本身扣掉了 prefill——而是同一条 prompt 反复跑时，缓存会让两次测量的起点不一致，加进 20–40% 的噪声里不划算。

### T3 KV 量化（1 小时）
```bash
python -m edgebench sweep --model <模型> --batches 1,8 --kv-dtype fp8 --max-tokens 256 --out runs/sweep-kvfp8.jsonl
```
要和 T2 的 b1/b8 逐点对照。同 batch、同 ctx，只差 kv-dtype。
注意：vLLM 的 `kv_cache_dtype` 只影响 KV 存储精度，模型权重不动。

### T4 MoE 卸载曲线（2 小时，**这是差异化重点**）
```bash
# 先确认卡上装不下：nvidia-smi 看显存，模型比自己算
python -m edgebench offload --bin-dir /root/llama.cpp/build/bin --model <moe.gguf> \
    --points 0,8,16,24,32,48 --n-gen 128 --reps 3
```
- 每个点记录 `gen_tok_s`，顺带 `nvidia-smi` 采一次显存占用。
- 要回答：**卸载多少层专家之后开始断崖**，断崖前那段"多放进去一点就多快一点"的区间有多宽。
- 如果所有点都 OOM，说明模型选小了，换更大的；如果一个点就撑满显存，说明选大了，换更小的。

### T5 精度回归（1 小时）
```bash
python -m edgebench tasks --engine vllm --model <模型> --tasks configs/tasks.jsonl --out runs/tasks-base.json
python -m edgebench ppl --bin-dir /root/llama.cpp/build/bin --model <同权重.gguf> --corpus <一段纯文本> --out runs/ppl-base.json
# 然后对每组"提速配置"（kv fp8 / 量化 / 卸载）重跑，用 judge() 对比
```
- 通过标准：PPL 涨 ≤2%，任务集掉 ≤1 题。
- 不通过就说明这个提速是拿精度换的，报告里必须写明。

## 6. 结果目录与字段

```
runs/
  env.json              # 卡、驱动、PCIe、版本
  sweep-*.jsonl         # 每行 {engine,model,batch,step_ms,ms_per_token,tok_s,kv_dtype,load_s}
  offload-*.json        # [{n_cpu_moe,ok,gen_tok_s,step_ms,...}]
  tasks-*.json          # {passed,total,rate,detail[]}
  ppl-*.json            # {ok,ppl}
```

对比用的基线永远放在 `runs/baseline-*.json`，改了什么就复制一份命名为 `runs/<改动>-*.json`。

## 7. 报告模板

```markdown
# <卡型> 推理性能实测 <日期>
环境: runs/env.json 摘要（卡、显存、驱动、PCIe、torch、vllm 版本）

## 1 吞吐
| batch | step_ms | tok/s | 离散度 | 理论预测 | 偏差 |
（下面一句话说拐点在哪）

## 2 KV 量化
| batch | fp16 tok/s | fp8 tok/s | 收益 | 精度代价 |
（一句话：值不值）

## 3 MoE 卸载
| n_cpu_moe | tok/s | 显存占用 | 相对满血 |
（一句话：断崖在哪个点，窗口多宽）

## 4 精度
| 配置 | PPL | Δ% | 任务通过 | Δ题 |
（一句话：哪些提速是白拿的，哪些是拿精度换的）

## 5 结论
这台卡上最值得做的一件事：______
最不值得做的一件事：______
与交接文档第 2 节的既有事实哪里对不上：______
```

## 8. 预算与止损

- 预算 **8 GPU 小时**。超了就停，把已有结果交出来。
- 单条命令超过 40 分钟没输出就 kill，怀疑 OOM 或模型下载卡住。
- 下模型慢：优先 modelscope（`modelscope download --model <id> --local_dir <dir>`），其次 hf-mirror。
- 任何一步失败超过 2 次，跳过它并写进"未完成"，不要死磕。

## 9. 失败排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `CUDA out of memory` | 模型 + KV 超了 | 降 `--max-len`、降 `--gpu-util`、换更小的量化权重 |
| tok/s 不随 batch 涨 | 引擎没开 continuous batching，或 `enforce_eager` | 去掉 `--eager`，确认 vLLM 版本 ≥0.6 |
| `step_ms` 为负 | 长短两次测量抖动过大 | 加 `--max-tokens`（≥256），减少引擎初始化干扰 |
| llama-bench 找不到 | 没编译或路径不对 | `ls /root/llama.cpp/build/bin/llama-bench` |
| offload 全 OOM | 模型没选对 | 选一个"能装下一半"的模型，不是全装不下 |
| 数字抖动 >30% | 云主机多租户抢占 / 功耗墙 | 同点重跑 5 次取中位数，报告里写明抖动范围 |

## 10. 回传

打包 `runs/` + `REPORT-<日期>.md`：

```bash
tar czf edge-result-$(date +%Y%m%d).tar.gz runs REPORT-*.md
```

把 tar.gz 和那份 REPORT 一起交回。原始 jsonl 比结论更重要 —— 结论文档是给人看的，jsonl 是给成本模型吃的。
