#!/usr/bin/env node
/**
 * costmodel - LLM 推理算力成本模型
 *
 * 原则：只建模"搬多少字节、算多少次乘加"，以及这两者在给定硬件上的时间。
 * 不建模"哪个方案更好"——那是人的判断。
 *
 * 用法见 README.md
 */

const VERSION = "0.1.0";

// ---------------------------------------------------------------------------
// 预设：模型
// 参数来自公开配置；kvBytesAt16 是每 token 每序列的 KV 字节数（FP16 基准），
// 非标准注意力（MLA）用实测/公式值覆盖。
// ---------------------------------------------------------------------------
const MODELS = {
  "qwen3-1.7b": {
    label: "Qwen3-1.7B (dense)",
    paramsB: 1.7, activeB: 1.7,
    layers: 28, hidden: 2048, heads: 16, kvHeads: 8, headDim: 128,
    weightBits: 16, kvBits: 16,
  },
  "qwen3-4b": {
    label: "Qwen3-4B (dense)",
    paramsB: 4.0, activeB: 4.0,
    layers: 36, hidden: 2560, heads: 32, kvHeads: 8, headDim: 128,
    weightBits: 16, kvBits: 16,
  },
  "llama3-70b": {
    label: "Llama-3-70B (dense, GQA 8)",
    paramsB: 70, activeB: 70,
    layers: 80, hidden: 8192, heads: 64, kvHeads: 8, headDim: 128,
    weightBits: 16, kvBits: 16,
  },
  "olmoe-1b-7b": {
    label: "OLMoE-1B-7B (MoE 6.9B/1.3B, 64 专家 top-8)",
    paramsB: 6.9, activeB: 1.3,
    layers: 16, hidden: 2048, heads: 16, kvHeads: 16, headDim: 128,
    weightBits: 16, kvBits: 16,
  },
  "deepseek-v3": {
    label: "DeepSeek-V3 (MoE 671B/37B, MLA)",
    paramsB: 671, activeB: 37,
    layers: 61, hidden: 7168, heads: 128, kvHeads: 128, headDim: 128,
    weightBits: 8, kvBits: 16,
    // MLA: 每层每 token 只存 kv_lora(512) + rope(64) = 576 个数
    kvBytesAt16: 61 * 576 * 2,
  },
};

// ---------------------------------------------------------------------------
// 预设：硬件
// 带宽是标称值；有效带宽按 eff 折算（实测通常 70%–85%）。
// ---------------------------------------------------------------------------
const HARDWARE = {
  "rtx5060-laptop": {
    label: "RTX 5060 Laptop 8GB",
    bandwidthGBs: 448,   // 128-bit GDDR7 28Gbps 标称
    peakTFLOPS: 60,      // FP16 稠密估算（不含稀疏加速）
    vramGB: 8,
    usableRatio: 0.92,
  },
  "a100-80g": {
    label: "A100 80GB SXM",
    bandwidthGBs: 2039, peakTFLOPS: 312, vramGB: 80, usableRatio: 0.95,
  },
  "h100-sxm": {
    label: "H100 SXM 80GB",
    bandwidthGBs: 3350, peakTFLOPS: 990, vramGB: 80, usableRatio: 0.95,
  },
};

// 能耗常数（只用于相对比较，绝对值误差很大）
const PJ_PER_BYTE = 20;      // HBM 读功耗约 20 pJ/byte
const PJ_PER_FLOP = 0.35;    // 稠密 FP16 约 0.35 pJ/flop

// ---------------------------------------------------------------------------

const USAGE = `costmodel ${VERSION}

用法: costmodel [选项]

  --model <name>     模型预设: ${Object.keys(MODELS).join(" | ")}   (默认 qwen3-1.7b)
  --hw <name>        硬件预设: ${Object.keys(HARDWARE).join(" | ")}   (默认 rtx5060-laptop)
  --ctx <n>          上下文长度 (默认 4096)
  --batch <n>        批大小 (默认 1)
  --wbits <n>        权重量化位宽 (默认取模型预设)
  --kvbits <n>       KV 量化位宽 (默认取模型预设)
  --spec <k>         投机解码草稿 token 数 (默认 0)
  --alpha <0-1>      草稿接受率 (默认 0.7)
  --bwEff <0-1>      有效带宽系数 (默认 0.8)
  --t0 <ms>          每步固定开销，kernel 启动/采样/调度 (默认 0)
  --flopsEff <0-1>   有效算力系数 (默认 0.4)
  --prompt <n>       额外的 prefill 预估：一次算完 n 个 prompt token (默认不算)
  --pfEff <0-1>      prefill 的有效算力系数 (默认 0.60，5060 上 dense 实测 0.57–0.60、MoE 0.37)
  --pft0 <ms>        prefill 的固定开销 (默认 6)
  --json             输出 JSON
  --selftest         跑自测
  --help             显示本帮助
`;

function parseArgs(argv) {
  const out = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const eq = a.indexOf("=");
      if (eq !== -1) {
        out[a.slice(2, eq)] = a.slice(eq + 1);
      } else if (argv[i + 1] && !argv[i + 1].startsWith("--")) {
        out[a.slice(2)] = argv[++i];
      } else {
        out[a.slice(2)] = true;
      }
    } else {
      out._.push(a);
    }
  }
  return out;
}

function pick(obj, key, fallback) {
  return key !== undefined && obj[key] !== undefined ? obj[key] : fallback;
}

function num(v, dflt) {
  if (v === undefined || v === true) return dflt;
  const n = Number(v);
  return Number.isFinite(n) ? n : dflt;
}

function kvBytesPerToken(model, kvBits) {
  const base = model.kvBytesAt16 !== undefined
    ? model.kvBytesAt16
    : model.layers * 2 * model.kvHeads * model.headDim * 2;
  return base * (kvBits / 16);
}

/**
 * 核心计算：一次 decode forward 的搬运量与计算量。
 * @param {object} model
 * @param {object} hw
 * @param {object} o { ctx, batch, wBits, kvBits, specK, specAlpha, bwEff, flopsEff, t0Ms }
 */
function evaluate(model, hw, o) {
  const ctx = o.ctx;
  const batch = o.batch;
  const k = o.specK;
  const alpha = o.specAlpha;
  const positions = batch * (1 + k);           // 这次 forward 处理的 token 位置数

  const weightBytes = model.activeB * 1e9 * (o.wBits / 8);
  const kvb = kvBytesPerToken(model, o.kvBits);
  const kvBytes = batch * (ctx + k) * kvb;     // 读旧 KV + 新写入
  const memBytes = weightBytes + kvBytes;

  // 2 * 参数 * 位置（linear 层）+ 4 * 层数 * 位置 * 上下文 * hidden（注意力两个矩阵乘）
  const flops = 2 * model.activeB * 1e9 * positions
    + 4 * model.layers * positions * ctx * model.hidden;

  const bw = hw.bandwidthGBs * 1e9 * o.bwEff;
  const tflops = hw.peakTFLOPS * 1e12 * o.flopsEff;

  const tMem = memBytes / bw;
  const tCmp = flops / tflops;
  const tStep = Math.max(tMem, tCmp) + (o.t0Ms || 0) / 1000;   // t0 = 每步固定开销（kernel 启动/采样/调度）

  const tokensOut = batch * (1 + alpha * k);   // 一次 forward 平均产出 token 数
  const msPerToken = (tStep * 1000) / tokensOut;
  const bytesPerToken = memBytes / tokensOut;
  const flopsPerToken = flops / tokensOut;

  return {
    ctx, batch, specK: k, specAlpha: alpha,
    wBits: o.wBits, kvBits: o.kvBits,
    weightBytes, kvBytes, memBytes,
    flops, positions, tokensOut,
    tMem, tCmp, tStep,
    msPerToken, bytesPerToken, flopsPerToken,
    tokensPerSec: 1000 / msPerToken,
    bound: tMem >= tCmp ? "mem" : "compute",
    memShare: kvBytes / memBytes,
    bwEff: o.bwEff, flopsEff: o.flopsEff,
    energyPerTokenJ: (bytesPerToken * PJ_PER_BYTE + flopsPerToken * PJ_PER_FLOP) * 1e-12,
  };
}

/**
 * prefill：把 P 个 prompt token 一次算完。瓶颈是算力，不是带宽——
 * 权重只读一遍（摊到 P 个 token 上），但每个 token 都要过全部线性层。
 * 注意力是 O(P²)：P 大到一定程度会盖过线性项，那时"长上下文贵"才真正开始。
 * @param {object} o { prompt, wBits, kvBits, bwEff, pfEff, pfT0Ms }
 */
function prefillCost(model, hw, o) {
  const P = o.prompt;
  // MoE 在长 prompt 下专家并集趋近全部，权重读取按总参数算；dense 两者相等
  const bytesParams = Math.max(model.activeB, model.paramsB) * 1e9 * (o.wBits / 8);
  const kvBytes = P * kvBytesPerToken(model, o.kvBits);      // prefill 只写 KV，不读
  const linearFlops = 2 * model.activeB * 1e9 * P;           // 每个 token 只算自己命中的专家
  const attnFlops = 4 * model.layers * P * P * model.hidden;
  const flops = linearFlops + attnFlops;

  const bw = hw.bandwidthGBs * 1e9 * o.bwEff;
  // MoE 的 prefill 要跑一堆小 GEMM（每个 token 命中的专家不同），实测效率只有 dense 的 2/3
  const isMoE = model.paramsB > model.activeB * 1.5;
  const effTFLOPS = hw.peakTFLOPS * o.pfEff * (isMoE ? 0.65 : 1);
  const tflops = effTFLOPS * 1e12;
  const tMem = bytesParams / bw;
  const tCmp = flops / tflops;
  const t = Math.max(tMem, tCmp) + (o.pfT0Ms || 0) / 1000;
  const msPerToken = (t * 1000) / P;

  return {
    prompt: P, bytesParams, kvBytes, linearFlops, attnFlops, flops,
    attnShare: attnFlops / flops,
    isMoE, effTFLOPS,
    tMem, tCmp, t,
    msPerToken,
    tokensPerSec: 1000 / msPerToken,
    bound: tMem >= tCmp ? "mem" : "compute",
    // 注意力项追平线性项的 prompt 长度
    attnCrossover: (2 * model.activeB * 1e9) / (4 * model.layers * model.hidden),
  };
}

function fmtBytes(b) {
  const gb = b / 1e9;
  if (gb >= 1) return gb.toFixed(1) + " GB";
  return (b / 1e6).toFixed(1) + " MB";
}

function fmtMs(ms) {
  if (ms >= 1000) return (ms / 1000).toFixed(2) + " s";
  if (ms >= 1) return ms.toFixed(1) + " ms";
  return ms.toFixed(2) + " ms";
}

function pad(s, w) {
  s = String(s);
  return s.length >= w ? s : s + " ".repeat(w - s.length);
}

function table(rows, headers) {
  const widths = headers.map((h, i) =>
    Math.max(h.length, ...rows.map(r => String(r[i]).length)));
  const line = (cells) => cells.map((c, i) => pad(c, widths[i])).join("  ");
  const out = [line(headers), widths.map(w => "-".repeat(w)).join("  ")];
  for (const r of rows) out.push(line(r));
  return out.join("\n");
}

function header(t) {
  return "\n" + t + "\n" + "=".repeat(Math.max(12, t.length));
}

function vramCheck(model, hw, wBits) {
  const need = model.paramsB * 1e9 * (wBits / 8);   // 常驻要按总参数算（MoE 也是）
  const cap = hw.vramGB * 1e9 * hw.usableRatio;
  if (need <= cap) return null;
  const rawBits = (cap / (model.paramsB * 1e9)) * 8;
  if (rawBits < 2) {
    return `权重 ${fmtBytes(need)} > 可用显存 ${fmtBytes(cap)}；`
      + `${model.label} 连 2bit 都装不下，需要换更小的模型或更多显存`;
  }
  const fitBits = Math.floor(rawBits * 2) / 2;
  return `权重 ${fmtBytes(need)} > 可用显存 ${fmtBytes(cap)}；`
    + `${model.label} 至少要 ${fitBits} bit 权重才装得下`;
}

function analysis(model, hw, base) {
  const kvb = kvBytesPerToken(model, base.kvBits);
  const wb = base.weightBytes;
  const lines = [];

  // KV 超过权重的上下文长度（batch=1）
  const ctxKvEqW = wb / kvb;
  lines.push(`KV 读量与权重持平的位置：ctx ≈ ${Math.round(ctxKvEqW).toLocaleString()}` +
    `（batch=1 时，超过它 KV 就是大头）`);

  // 单序列算力/带宽拐点：算 B* 使两边相等
  const perTokenFlops = 2 * model.activeB * 1e9 + 4 * model.layers * base.ctx * model.hidden;
  const bw = hw.bandwidthGBs * 1e9 * base.bwEff;
  const tflops = hw.peakTFLOPS * 1e12 * base.flopsEff;
  const perSeqBytes = (base.ctx + base.specK) * kvb;
  const denom = (bw / tflops) * perTokenFlops - perSeqBytes;
  if (denom > 0) {
    const bStar = wb / denom;
    lines.push(`batch 拐点：B* ≈ ${Math.round(bStar)}（超过这个批大小，瓶颈从显存带宽转到算力）`);
  } else {
    lines.push("batch 拐点：不存在（吞吐永远受显存带宽限制）");
  }

  if (base.specK > 0) {
    const speedup = (1 + base.specAlpha * base.specK) / (1 + base.specK);
    lines.push(`投机解码：算力受限时理论加速比 ${speedup.toFixed(2)}×（<1 就是纯亏），` +
      `带宽受限时约为 ${((1 + base.specAlpha * base.specK)).toFixed(2)}×`);
  }

  const vram = vramCheck(model, hw, base.wBits);
  if (vram) lines.push("显存：" + vram);

  return lines;
}

function compareRows(model, hw, o) {
  const variants = [];

  const run = (name, patch) => {
    const cfg = { ...o, ...patch };
    const r = evaluate(model, hw, cfg);
    variants.push({ name, r, cfg });
  };

  run("baseline", {});
  if (o.wBits === 16) {
    run("weights fp8", { wBits: 8 });
    run("weights int4", { wBits: 4 });
  } else if (o.wBits === 8) {
    run("weights int4", { wBits: 4 });
  }
  if (o.kvBits === 16) {
    run("kv fp8", { kvBits: 8 });
    run("kv int4", { kvBits: 4 });
  }
  run(`batch x${o.batch * 8}`, { batch: o.batch * 8 });
  run(`ctx x8 (${Math.min(o.ctx * 8, 131072)})`, { ctx: Math.min(o.ctx * 8, 131072) });
  run("spec k=3 a=.6", { specK: 3, specAlpha: 0.6 });

  const all = { wBits: o.wBits > 4 ? 4 : o.wBits, kvBits: Math.min(o.kvBits, 8), batch: o.batch * 8, specK: 3, specAlpha: 0.6 };
  run("all combined", all);

  return variants;
}

// ---------------------------------------------------------------------------

function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help) {
    console.log(USAGE);
    return;
  }
  if (args.selftest) return selftest();

  const modelKey = args.model || "qwen3-1.7b";
  const hwKey = args.hw || "rtx5060-laptop";
  const model = MODELS[modelKey];
  const hw = HARDWARE[hwKey];
  if (!model) {
    console.error(`未知模型: ${modelKey}；可选: ${Object.keys(MODELS).join(", ")}`);
    process.exit(2);
  }
  if (!hw) {
    console.error(`未知硬件: ${hwKey}；可选: ${Object.keys(HARDWARE).join(", ")}`);
    process.exit(2);
  }

  const o = {
    ctx: num(args.ctx, 4096),
    batch: num(args.batch, 1),
    wBits: num(args.wbits, model.weightBits),
    kvBits: num(args.kvbits, model.kvBits),
    specK: num(args.spec, 0),
    specAlpha: num(args.alpha, 0.7),
    bwEff: num(args.bwEff, 0.8),
    flopsEff: num(args.flopsEff, 0.4),
    t0Ms: num(args.t0, 0),
    prompt: num(args.prompt, 0),
    pfEff: num(args.pfEff, 0.60),
    pfT0Ms: num(args.pft0, 6),
  };

  const base = evaluate(model, hw, o);

  if (args.json) {
    const rows = compareRows(model, hw, o).map(v => ({
      name: v.name, cfg: v.cfg,
      msPerToken: v.r.msPerToken, bytesPerToken: v.r.bytesPerToken,
      bound: v.r.bound, energyPerTokenJ: v.r.energyPerTokenJ,
    }));
    console.log(JSON.stringify({
      version: VERSION, model: modelKey, hw: hwKey,
      input: o, base, variants: rows,
      analysis: analysis(model, hw, base),
    }, null, 2));
    return;
  }

  const lines = [];
  lines.push(`costmodel ${VERSION}`);
  lines.push(`模型 ${model.label}   激活 ${model.activeB}B / 总 ${model.paramsB}B`);
  lines.push(`硬件 ${hw.label}   标称 ${hw.bandwidthGBs} GB/s · ${hw.peakTFLOPS} TFLOPS`);
  lines.push(`输入 ctx=${o.ctx} batch=${o.batch} w=${o.wBits}bit kv=${o.kvBits}bit` +
    (o.specK ? ` spec k=${o.specK} a=${o.specAlpha}` : ""));

  lines.push(header("单次 decode forward"));
  lines.push(`权重读取        ${fmtBytes(base.weightBytes)}`);
  lines.push(`KV 读取+写入    ${fmtBytes(base.kvBytes)}  (占搬运量 ${(base.memShare * 100).toFixed(1)}%)`);
  lines.push(`搬运总计        ${fmtBytes(base.memBytes)}`);
  lines.push(`计算量          ${(base.flops / 1e9).toFixed(1)} GFLOP`);
  lines.push(`显存受限耗时    ${fmtMs(base.tMem * 1000)}`);
  lines.push(`算力受限耗时    ${fmtMs(base.tCmp * 1000)}`);
  lines.push(`瓶颈            ${base.bound}`);
  lines.push(`每 token        ${fmtMs(base.msPerToken)}  (${base.tokensPerSec.toFixed(1)} tok/s，理论上限)`);
  lines.push(`每 token 能耗   ${(base.energyPerTokenJ * 1000).toFixed(2)} mJ（估算，不含静态功耗，仅相对有意义）`);

  if (o.prompt > 0) {
    const pf = prefillCost(model, hw, o);
    lines.push(header(`prefill：一次吃 ${o.prompt} 个 prompt token`));
    lines.push(`权重读取        ${fmtBytes(pf.bytesParams)}（只读一遍，摊到每个 token）`);
    lines.push(`KV 写入         ${fmtBytes(pf.kvBytes)}（prefill 不读旧 KV）`);
    lines.push(`计算量          ${(pf.flops / 1e9).toFixed(1)} GFLOP（其中注意力占 ${(pf.attnShare * 100).toFixed(1)}%）`);
    lines.push(`有效算力        ${pf.effTFLOPS.toFixed(1)} TFLOPS（标称 ${hw.peakTFLOPS}，${pf.isMoE ? "MoE 小 GEMM 打折" : "dense"}）`);
    lines.push(`显存受限耗时    ${fmtMs(pf.tMem * 1000)}`);
    lines.push(`算力受限耗时    ${fmtMs(pf.tCmp * 1000)}`);
    lines.push(`瓶颈            ${pf.bound}`);
    lines.push(`平均每 token    ${fmtMs(pf.msPerToken)}  (${pf.tokensPerSec.toFixed(1)} tok/s，理论上限)`);
    lines.push(`注意力追平线性  P ≈ ${Math.round(pf.attnCrossover).toLocaleString("en-US")} token 之后，长上下文问题才真正开始`);
  }

  lines.push(header("对比：每个杠杆单独看"));
  const variants = compareRows(model, hw, o);
  const rows = variants.map(v => [
    v.name,
    fmtBytes(v.r.bytesPerToken),
    fmtMs(v.r.msPerToken),
    v.r.tokensPerSec.toFixed(1),
    (base.msPerToken / v.r.msPerToken).toFixed(2) + "x",
    v.r.bound,
  ]);
  lines.push(table(rows, ["变体", "字节/token", "ms/token", "tok/s", "相对baseline", "瓶颈"]));
  lines.push("");
  lines.push("说明：'x8' 两行改的是工作负载，不是优化；'all combined' 是纯带宽模型的极限，");
  lines.push("      实际会被 kernel 启动、调度、通信吃掉一部分，别当承诺");

  lines.push(header("结构分析"));
  for (const l of analysis(model, hw, base)) lines.push("- " + l);

  lines.push(header("假设"));
  lines.push(`- 有效带宽 ${(o.bwEff * 100).toFixed(0)}%，有效算力 ${(o.flopsEff * 100).toFixed(0)}%${o.t0Ms ? `，每步固定开销 ${o.t0Ms} ms` : ""}`);
  lines.push(`- 能耗常数 ${PJ_PER_BYTE} pJ/byte, ${PJ_PER_FLOP} pJ/flop`);
  lines.push("- 不含 kernel 启动、调度、通信开销；不含 prefill");
  lines.push("- 相对结论对有效系数不敏感，绝对值敏感");

  console.log(lines.join("\n"));
}

function selftest() {
  const model = MODELS["qwen3-1.7b"];
  const hw = HARDWARE["rtx5060-laptop"];
  const base = evaluate(model, hw, {
    ctx: 4096, batch: 1, wBits: 16, kvBits: 16, specK: 0, specAlpha: 0.7,
    bwEff: 0.8, flopsEff: 0.4,
  });

  const checks = [];
  const check = (name, cond, detail) => checks.push({ name, ok: !!cond, detail });

  check("权重字节 = 参数*2", Math.abs(base.weightBytes - 1.7e9 * 2) < 1,
    `${base.weightBytes}`);

  const kvb = kvBytesPerToken(model, 16);
  check("KV 字节 = 28层*2*8头*128维*2B", kvb === 28 * 2 * 8 * 128 * 2, `${kvb}`);

  const dbl = evaluate(model, hw, {
    ctx: 8192, batch: 1, wBits: 16, kvBits: 16, specK: 0, specAlpha: 0.7,
    bwEff: 0.8, flopsEff: 0.4,
  });
  check("ctx 翻倍 → KV 翻倍", Math.abs(dbl.kvBytes - base.kvBytes * 2) < 1,
    `${base.kvBytes} -> ${dbl.kvBytes}`);

  const q = evaluate(model, hw, {
    ctx: 4096, batch: 1, wBits: 4, kvBits: 16, specK: 0, specAlpha: 0.7,
    bwEff: 0.8, flopsEff: 0.4,
  });
  check("int4 权重 = 1/4", Math.abs(q.weightBytes - base.weightBytes / 4) < 1,
    `${q.weightBytes}`);

  const spec = evaluate(model, hw, {
    ctx: 4096, batch: 1, wBits: 16, kvBits: 16, specK: 3, specAlpha: 0.6,
    bwEff: 0.8, flopsEff: 0.4,
  });
  check("投机解码 3+0.6*3 个 token/forward",
    Math.abs(spec.tokensOut - 2.8) < 1e-9, `${spec.tokensOut}`);

  check("单 token 时间 = 字节/带宽（带宽受限时）",
    Math.abs(base.msPerToken - base.memBytes / (448e9 * 0.8) * 1000) < 1e-6,
    `${base.msPerToken}`);

  let failed = 0;
  for (const c of checks) {
    console.log(`${c.ok ? "PASS" : "FAIL"}  ${c.name}  ${c.ok ? "" : "-> " + c.detail}`);
    if (!c.ok) failed++;
  }
  console.log(`\n${checks.length - failed}/${checks.length} passed`);
  process.exit(failed ? 1 : 0);
}

main();
