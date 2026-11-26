#!/usr/bin/env node
// 量 prefill 成本：不同 prompt 长度下的 prompt_eval 吞吐
// 用法: node prefill.mjs [model] [--lens 128,512,2048,8192] [--runs 3] [--json]

const args = process.argv.slice(2);
const opt = { lens: [128, 512, 2048, 8192], runs: 3, json: false, model: "qwen3-1.7b", host: "http://127.0.0.1:11434" };
const positional = [];
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === "--json") opt.json = true;
  else if (a === "--lens") opt.lens = args[++i].split(",").map(Number);
  else if (a === "--runs") opt.runs = Number(args[++i]);
  else if (a === "--host") opt.host = args[++i];
  else if (a === "--numctx") opt.numctx = Number(args[++i]);
  else positional.push(a);
}
if (positional[0]) opt.model = positional[0];

// 用英文重复段落，tokenizer 下接近 1 词 ≈ 1.3 token，实际长度由服务端回传的 prompt_eval_count 决定
const PARA = "Memory bandwidth determines the floor of decode latency, while capacity determines the ceiling of what you can run at once. ";
function makePrompt(targetTokens) {
  const words = PARA.split(" ");
  const out = [];
  for (let i = 0; i < Math.ceil(targetTokens * 0.8); i++) out.push(words[i % words.length]);
  return out.join(" ");
}

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

async function gen(prompt, numCtx) {
  // Ollama 会缓存 prompt 前缀，重复的前缀会让 prompt_eval_duration 趋近 0。
  // 每次在开头加一个唯一 nonce（对前缀做扰动），保证真的走完整 prefill。
  const nonce = "nonce " + Math.random().toString(36).slice(2, 10) + " ";
  const body = {
    model: opt.model,
    prompt: nonce + prompt,
    stream: false,
    options: { num_predict: 1, temperature: 0, num_ctx: numCtx },
  };
  const r = await fetch(opt.host + "/api/generate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r.json();
}

const rows = [];
for (const target of opt.lens) {
  const prompt = makePrompt(target);
  const numCtx = opt.numctx ? opt.numctx : Math.max(4096, Math.ceil(target * 1.6) + 256);
  await gen(prompt, numCtx); // warmup
  const ts = [];
  let tokens = 0;
  for (let i = 0; i < opt.runs; i++) {
    const j = await gen(prompt, numCtx);
    tokens = j.prompt_eval_count || 0;
    const dur = (j.prompt_eval_duration || 0) / 1e9;
    if (dur > 0) ts.push(tokens / dur);
  }
  const t = median(ts);
  rows.push({ target, prompt_tokens: tokens, prefill_tok_s: Math.round(t * 10) / 10 });
  if (!opt.json) console.log(`prompt ${String(tokens).padStart(5)} token   prefill ${String(Math.round(t * 10) / 10).padStart(8)} tok/s   ${(1000 / t).toFixed(2)} ms/token`);
}

if (opt.json) console.log(JSON.stringify({ model: opt.model, rows }, null, 2));
