# costmodel

LLM 推理算力成本模型。给定模型规模和硬件，算出**每生成一个 token 要搬多少字节、算多少次乘加**，以及在给定硬件上这两者谁先撞墙。

它是模型，不是实测。数字的用途是比大小、找拐点，不是报产能。

## 不是什么

- 不是 benchmark，不跑任何模型
- prefill 与 decode 分开算：decode 默认算，prefill 加 `--prompt <n>`（瓶颈不同，有效算力也不同）
- 不含 kernel 启动、调度、通信、MoE 路由开销
- 能耗是相对量，绝对值别信

## 快速开始

```bash
cd D:\deepseek\llm-cost-lab\costmodel
costmodel.cmd                     # 默认：Qwen3-1.7B + 本机 5060
costmodel.cmd --model llama3-70b
costmodel.cmd --model deepseek-v3 --wbits 8 --ctx 32768
costmodel.cmd --prompt 8192       # 额外算一次 8192 token 的 prefill
costmodel.cmd --json              # 给脚本吃
costmodel.cmd --selftest          # 6 条不变量检查
```

## 公式（可以复算）

一次 decode forward，批大小 B、上下文 L、投机草稿 k 个：

```
权重搬运   = 激活参数量 × 位宽/8
KV 搬运    = B × (L + k) × 每tokenKV字节
总搬运     = 权重 + KV
计算量     = 2 × 激活参数 × P + 4 × 层数 × P × L × hidden     (P = B×(1+k))
记忆耗时   = 总搬运 / 有效带宽
算力耗时   = 计算量 / 有效算力
单步耗时   = max(记忆, 算力)
每token    = 单步耗时 / (B × (1 + αk))
```

每 token KV 字节（FP16）= `层数 × 2 × KV头数 × 头维度 × 2`。MLA 这类非标准注意力直接用 `kvBytesAt16` 覆盖。

## 结构分析里的三个数

| 输出 | 含义 |
|---|---|
| KV 拐点 | 上下文多长时，KV 读量超过权重读量。过了它，省 KV 比省权重更值钱 |
| batch 拐点 | 批多大时瓶颈从带宽转到算力。过了它，量化权重不再提速，投机解码开始亏 |
| 显存检查 | 权重放不下时给出最低位宽；连 2bit 都装不下就直接说装不下，不编数 |

## 参数来源与不确定度

| 参数 | 来源 | 不确定度 |
|---|---|---|
| 参数量 / 层数 / 头数 | 公开配置 | 低 |
| 标称带宽 448 GB/s | 128-bit GDDR7 28Gbps 推算 | 中，实测通常 70%–85% |
| 60 TFLOPS | FP16 稠密估算 | 高，可能差 30% |
| 20 pJ/byte、0.35 pJ/flop | 工艺量级常数 | 高，只用于相对比较 |

**相对结论对有效系数不敏感**（都是线性缩放时相消），绝对值敏感。要改系数：`--bwEff 0.7 --flopsEff 0.3`。

## 本机跑出来的数（RTX 5060 Laptop 8GB）

Qwen3-1.7B，FP16，ctx 4096，batch 1：

| 变体 | 字节/token | ms/token | 相对 baseline |
|---|---|---|---|
| baseline | 3.9 GB | 10.8 | 1.00× |
| 权重 FP8 | 2.2 GB | 6.1 | 1.78× |
| 权重 INT4 | 1.3 GB | 3.7 | 2.93× |
| KV FP8 | 3.6 GB | 10.1 | 1.06× |
| batch ×8 | 894 MB | 2.5 | 4.32× |
| 投机 k=3 α=0.6 | 1.4 GB | 3.9 | 2.80× |

三个直接结论：

1. **量化在 decode 阶段接近线性提速**，不是"按比例温和降功耗"——因为瓶颈是读权重，位宽减半读量就减半
2. **短上下文量化 KV 几乎没用**（1.06×），KV 拐点在 3 万 token 左右；长上下文才轮到它
3. **batch 是最便宜的一刀**（8 倍共享一次权重读取），但拐点之后转成算力瓶颈

对照（都按 FP16 权重）：Llama-3-70B 单 token 要搬 141 GB，本机 394 ms/token，batch 拐点 154——这张卡上跑不了，数字只说明规模差。DeepSeek-V3（37B 激活 + MLA）是 74 GB，KV 拐点到 105 万 token 才出现，MLA 的意义就在这。同精度下，MoE 把每 token 的搬运量压到 dense 70B 的一半左右，靠的不是 KV 而是激活比。

## 实测校准（2026-11-22，RTX 5060 Laptop 8GB）

Ollama 0.40.16 真跑，`calibrate.mjs` 每次 warmup + 5 次取中位数，每 token 时间取 `eval_duration / eval_count`：

| 模型 | 权重大小 | 序列长度 | ms/token | tok/s | 反解 bwEff |
|---|---|---|---|---|---|
| Qwen3-1.7B Q4_K_M | 1.11 GB | ~330 | 5.30 | 188.8 | 0.47 |
| Qwen3-1.7B Q8_0 | 1.83 GB | ~300 | 7.67 | 130.3 | 0.53 |
| Qwen3-1.7B Q4_K_M | 1.11 GB | 6,570 | 7.45 | 134.3 | — |
| Qwen3-4B Q4_K_M | 2.50 GB | ~250 | 10.73 | 93.2 | 0.52 |
| OLMoE-1B-7B Q4_K_M（MoE） | 4.21 GB / 实读 0.78 GB | ~400 | 4.10 | 244.0 | 0.67* |

\* MoE 反解要用每 token 实读的字节（1.3B 激活 × 4.5bit ≈ 0.73 GB + KV），不是文件大小；
用 `calibrate.mjs --bytes 0.78` 覆盖。

MoE 这一行是对"一次前向只碰激活参数"的直接验证：文件 4.21 GB，每 token 只读 0.78 GB，
和同样 4.2 GB 的 dense 7B（按上表斜率推算约 68 tok/s）比，快 3.6 倍；比小它 3.8 倍的 Qwen3-1.7B 还快 1.3 倍。
代价是 4.2 GB 权重必须全装进显存——用显存换带宽。

两种独立的差分都对上了同一个结论：

- **量化的差分**：同一个模型，Q8 权重比 Q4 多 1.65 倍，慢 1.45 倍 → 边际 304 GB/s
- **上下文的差分**：序列从 67 拉到 6,570，多搬 745 MB KV，多花 2.15 ms → 边际 343 GB/s

所以 `t ≈ 1.5 ms + bytes / 320 GB/s`（有效带宽是标称 448 的 56–76%）。
小模型慢不是因为带宽差，是因为那 1.5 ms 固定开销（kernel 启动、采样、Go 层调度）摊不开——
1.7B 每 token 理论搬运只要 3.5 ms，固定开销占了三分之一。

按这个拟合，跑大模型时用 `--bwEff 0.7 --t0 1.5`。

注意同一条件重复跑会差 20–40%（笔记本的功耗墙和温度），校准精度就到这个量级，
别把小数点后第二位当回事。上面第一版数据里有几个 125 tok/s 的点就是这么来的，已被中位数替代。

## 怎么校准成实测

```
bwEff = 总搬运 / (实测 ms/token × 标称带宽)
```

`calibrate.mjs` 直接干这个：连 Ollama，跑 warmup + N 次生成，用 `eval_count/eval_duration`
算 tok/s，反解出 `--bwEff`。`longctx.mjs` 量长上下文的 KV 边际成本。

## 已知局限

- prefill 只在单卡单机上拟合过（5060：dense 57–60%、MoE 37%），换卡要重新标定
- prefill 只测到 7.2k token，更长 prompt 的曲线是外推
- 无 PD 分离：prefill 和 decode 抢同一张卡的成本没算
- 无通信：多卡 EP/TP 的 all-to-all 开销不在内
- 固定开销：`--t0` 是常数近似，真机上它跟 batch、序列长度都有关
- MoE 只按激活参数算搬运，不建模专家并行下的负载不均

## 版本

0.1 — decode 单步模型、三个预设、对比表、自测
0.2 — 加 `--t0` 固定开销项；附 Ollama 实测校准
0.3 — 加 prefill（`--prompt`），decode/prefill 两套有效算力分开；补 `qwen3-4b` 预设
下一步：PD 分离、稀疏注意力
