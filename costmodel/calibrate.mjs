#!/usr/bin/env node
/**
 * calibrate - 用 Ollama 实测反解 costmodel 的有效带宽系数
 *
 * 数据全部来自 Ollama 自己的计时（eval_count / eval_duration），
 * 不在本机掐表，避开进程启动和网络抖动。
 */

const BASE = process.env.OLLAMA_HOST || "http://127.0.0.1:11434";
const NOMINAL_BW_GBS = 448;   // 与 costmodel 的 rtx5060-laptop 预设一致

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const eq = a.indexOf("=");
      if (eq !== -1) out[a.slice(2, eq)] = a.slice(eq + 1);
      else if (argv[i + 1] && !argv[i + 1].startsWith("--")) out[a.slice(2)] = argv[++i];
      else out[a.slice(2)] = true;
    }
  }
  return out;
}

async function api(path, body) {
  const res = await fetch(BASE + path, {
    method: body ? "POST" : "GET",
    headers: { "content-type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${path} -> HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const model = args.model || "qwen3:1.7b";
  const runs = Number(args.runs || 3);
  const ctx = Number(args.ctx || 4096);
  const predict = Number(args.predict || 256);

  let tags;
  try {
    tags = await api("/api/tags");
  } catch (e) {
    console.error(`连不上 Ollama (${BASE})：${e.message}`);
    console.error("先跑: ollama serve");
    process.exit(2);
  }

  const info = (tags.models || []).find(m => m.name === model || m.name === model + ":latest");
  if (!info) {
    console.error(`本地没有 ${model}。已装：${(tags.models || []).map(m => m.name).join(", ") || "(空)"}`);
    console.error(`先跑: ollama pull ${model}`);
    process.exit(2);
  }

  // MoE 每 token 只读激活的专家，用 --bytes 覆盖（单位 GB）
  const sizeBytes = args.bytes ? Number(args.bytes) * 1e9 : info.size;
  const prompt = "用三百字说明：为什么大模型 decode 阶段主要受显存带宽而不是算力限制。";

  const body = {
    model,
    prompt,
    stream: false,
    think: false,
    options: { num_ctx: ctx, num_predict: predict, temperature: 0 },
  };

  console.log(`model ${model}  size ${(sizeBytes / 1e9).toFixed(2)} GB  ctx ${ctx}  predict ${predict}`);

  // 第一次会加载模型，不计入
  process.stdout.write("warmup... ");
  const warm = await api("/api/generate", body);
  console.log(`${(warm.eval_count / (warm.eval_duration / 1e9)).toFixed(1)} tok/s (load ${(warm.load_duration / 1e9).toFixed(1)}s)`);

  const tps = [];
  const prefillTps = [];
  for (let i = 0; i < runs; i++) {
    const r = await api("/api/generate", body);
    const t = r.eval_count / (r.eval_duration / 1e9);
    const p = r.prompt_eval_count / (r.prompt_eval_duration / 1e9);
    tps.push(t);
    prefillTps.push(p);
    console.log(`run ${i + 1}: ${t.toFixed(1)} tok/s  (${r.eval_count} tokens)  prefill ${p.toFixed(0)} tok/s`);
  }

  const med = median(tps);
  const msPerToken = 1000 / med;

  // 短上下文下 KV 搬运可以忽略，字节/token ≈ 权重文件大小
  const kvPerToken = Number(args.kvBytes || 0);
  const bytesPerToken = sizeBytes + kvPerToken * ctx;
  const bwEff = bytesPerToken / (msPerToken / 1000) / (NOMINAL_BW_GBS * 1e9);

  console.log("");
  console.log(`median        ${med.toFixed(1)} tok/s  =  ${msPerToken.toFixed(2)} ms/token`);
  console.log(`prefill       ${median(prefillTps).toFixed(0)} tok/s`);
  console.log(`bytes/token   ${(bytesPerToken / 1e9).toFixed(2)} GB`);
  console.log(`反解 bwEff    ${bwEff.toFixed(2)}   (标称 ${NOMINAL_BW_GBS} GB/s)`);
  console.log("");
  console.log(`喂回 costmodel:  --bwEff ${bwEff.toFixed(2)}`);
  if (bwEff > 0.95) {
    console.log("警告: bwEff > 0.95，说明标称带宽偏低或模型太小（kernel 开销占比高），换个更大的模型再测");
  }
}

main().catch(e => {
  console.error(e.message);
  process.exit(1);
});
