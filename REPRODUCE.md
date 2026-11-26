# 怎么复现这份报告

报告里的每个数字都有出处：脚本在 `costmodel/` 和 `edgebench/`，原始输出在 `edgebench/runs-4090/`。
这份文件按"你需要多少硬件"分三档，每档给命令、预期结果、耗时和花费。

**基准环境锁在**：vLLM 0.10.2 ／ transformers 4.55.4 ／ torch 2.8.0+cu128 ／ Python 3.10.8 ／
CUDA 11.8（编译 llama.cpp）／ Ollama 0.40.16 ／ Node 24。完整清单 `edgebench/requirements.txt`。
**换版本必须重跑校准**，别把 `801 GB/s`、`26.7×` 直接搬到新环境。

---

## 第 0 档：先确认包是完整的（10 秒）

```bash
python edgebench/selftest.py
```

期望：`脚本 16 个 | 数据 29 个 | 报告引用 15 项` + `通过 / 失败 0`。
它会检查四件事：脚本能否编译、JSON 能否解析、报告引用的文件在不在、costmodel 自测是否 6/6。

## 第 1 档前：没有 GPU 也能算（10 秒）

复现成本模型的公式与预测，不需要显卡。

```bash
cd costmodel
costmodel.cmd --selftest                    # 期望：6/6 passed
costmodel.cmd --model qwen3-1.7b --ctx 4096 # 期望：3.9 GB/token、10.8 ms、92.6 tok/s 上限
costmodel.cmd --model deepseek-v3 --wbits 8 --ctx 32768
costmodel.cmd --model qwen3-4b --ctx 4096 --prompt 8192   # 带 prefill
```

对应报告的：第 4 章（模型）、第 7.3 节（prefill 模型）。

---

## 第 2 档：一台 Windows + 任意 8 GB 显卡（约 40 分钟）

复现笔记本侧的四张表：权重带宽、固定开销、长上下文 KV、prefill。

```bash
# 1) 装 Ollama 0.40.16，模型放 D 盘（别塞 C 盘）
setx OLLAMA_MODELS "D:\ollama\models"
ollama create qwen3-1.7b -f Modelfile.qwen3

# 2) 反解带宽：期望 5 次中位数 ~188 tok/s，bwEff ≈ 0.47
node calibrate.mjs --model qwen3-1.7b

# 3) 长上下文 KV 边际成本：期望每 1000 token 多 0.19 ms，边际带宽 343 GB/s
node longctx.mjs

# 4) prefill 曲线：期望 dense 34–36 TFLOPS，MoE 22.5
node prefill.mjs qwen3-1.7b --lens 128,512,1024,2048,4096,8192 --runs 3 --numctx 16384
node prefill.mjs qwen3-4b    --lens 128,512,1024,2048,4096      --runs 3 --numctx 16384
```

对应报告的：5.2（拟合）、6.3（长上下文）、7.2（prefill）。

**偏差预期**：同一台机器连跑会热降频 10–15%。`prefill.mjs` 必须在开头塞唯一 nonce，
否则 Ollama 的前缀缓存会让 `prompt_eval_duration` 变成 0（这是附录 A 第 3 条坑）。

---

## 第 3 档：租一台 24 GB 卡（约 3 小时 / 十几块钱）

复现 4090 侧的全部实验。下面按报告的章节顺序排，跑完一份就有一份表。

### 准备

```bash
# 环境（Linux, Ubuntu 22.04）
pip install -r edgebench/requirements.txt
python -m edgebench doctor          # 期望全 OK；只认"能跑的 matmul"，不认 arch 列表

# 模型（国内走 ModelScope）
modelscope download --model Qwen/Qwen3-4B --local_dir ~/models/Qwen3-4B
modelscope download --model Qwen/Qwen3-30B-A3B-GPTQ-Int4 --local_dir ~/models/30B-A3B-GPTQ
```

### 6.2 dense 的 batch 曲线

```bash
python -m edgebench sweep --model ~/models/Qwen3-4B --batches 1,4,8,16,32
```
预期：10.14 ms 单步；98.6 → 2,635.8 tok/s（26.7×）；权重侧 801 GB/s。

### 6.3 长上下文与 fp8 KV

```bash
python -m edgebench sweep --model ~/models/Qwen3-4B --ctx-tokens 8192
python -m edgebench sweep --model ~/models/Qwen3-4B --ctx-tokens 8192 --kv-dtype fp8
```
预期：每千 token +0.19 ms；fp8 反而慢 21–26%，日志会打印
`Cannot use FlashAttention backend for FP8 KV cache.`

### 6.7 / 6.8 MoE 的 batch 与容量墙

```bash
python -m edgebench sweep --model ~/models/30B-A3B-GPTQ --batches 1,4,8,16,32
python -m edgebench sweep --model ~/models/30B-A3B-GPTQ --ctx-tokens 8192 --batches 1,4,8,16
```
预期：18.3×；日志 `GPU KV cache size: 50,160 tokens`；8k 时 batch 8 峰值 610 t/s、batch 16 崩到 526。

### 6.11 功耗

```bash
python edgebench/power_probe.py --model ~/models/Qwen3-4B --batches 1,4,8,16,32
```
预期：3,077 → 125 mJ/token；功率 283 → 301 W。

### 6.12 投机解码

```bash
python edgebench/spec_probe.py --model ~/models/Qwen3-4B --method ngram
```
预期：自由生成 1.05×，复述型 1.5–1.9×。

### 6.13 精度

```bash
# 语料：wikitext-2 test
curl -sL -o wt2.parquet "https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
python edgebench/ppl_probe.py --model ~/models/Qwen3-4B     --corpus wt2.txt --chunks 32 --out runs/ppl-bf16.json
python edgebench/ppl_probe.py --model ~/models/Qwen3-4B-AWQ --corpus wt2.txt --chunks 32 --out runs/ppl-awq.json
python edgebench/mc_probe.py  --model ~/models/Qwen3-4B     --out runs/mc-bf16.json
python edgebench/mc_probe.py  --model ~/models/Qwen3-4B-AWQ --out runs/mc-awq.json
```
预期：PPL 15.17 → 17.08；多选题 28/32 → 27/32。
（`ppl_check.py` 是交叉校验脚本，用 transformers 直接前向对比 vLLM 的 logprob，差 <0.5%。）

### 6.7 路由热度

```bash
python edgebench/router_probe.py --model ~/models/30B-A3B-GPTQ --batches 1,8,32
```
预期：并集 6.25% / 34.4% / 42.9%。
脚本内部已处理三个坑：`enforce_eager=True`、`VLLM_ENABLE_V1_MULTIPROCESSING=0`、固定 128 桶直方图。

### 6.4 MoE 卸载（要自己编 llama.cpp）

```bash
pip download llama-cpp-python --no-binary :all:      # 只为拿它 vendor 的完整源码树
tar xzf llama_cpp_python-0.3.38.tar.gz
cd llama_cpp_python-0.3.38/vendor/llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DLLAMA_CURL=OFF \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build -j 96 --target llama-bench
python -m edgebench offload --bin-dir build/bin --model <30B-A3B Q4_K_M gguf>
```
预期：`-ngl 99` 268.9 t/s；`--n-cpu-moe 1` 直接掉到 46.7（−83%）。

---

## 为什么你的数字会和报告不一样

| 偏差 | 量级 | 原因 |
|---|---|---|
| 笔记本连跑 | −10~15% | 功耗墙，prefill 比 decode 先撞 |
| 4090 同条件重跑 | ±5% | 宿主共享、时钟漂移 |
| 不同引擎横比 | 不可比 | GPTQ 与 GGUF 的反量化路径不同 |
| 换 vLLM 版本 | 可能很大 | 调度器实现变了，必须重跑 `calibrate` |

**唯一可靠的是同场比值**（如 26.7×、18.3×、−83%），绝对值只代表"这台机器这一天"。
