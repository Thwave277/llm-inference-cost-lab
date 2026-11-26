#!/usr/bin/env node
// 长上下文 KV 实测：固定 prompt 长度，只改生成长度，用 eval_* 数据算 tok/s
const BASE = "http://127.0.0.1:11434";

async function gen(model, prompt, numCtx, numPredict) {
  const res = await fetch(BASE + "/api/generate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model, prompt, stream: false, think: false,
      options: { num_ctx: numCtx, num_predict: numPredict, temperature: 0 },
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

const filler = "The quick brown fox jumps over the lazy dog and keeps running. ";
const model = process.argv[2] || "qwen3-1.7b";
const reps = Number(process.argv[3] || 500);

const long = filler.repeat(reps) + "\n\nReply with exactly three short sentences about bandwidth.";
const short = "Reply with exactly three short sentences about bandwidth.";

for (const [name, prompt, ctx] of [["short", short, 2048], ["long", long, 8192]]) {
  await gen(model, prompt, ctx, 8);            // warm
  const r = await gen(model, prompt, ctx, 64);
  const seq = r.prompt_eval_count + r.eval_count;
  const t = r.eval_count / (r.eval_duration / 1e9);
  const pref = r.prompt_eval_count / (r.prompt_eval_duration / 1e9);
  console.log(`${name.padEnd(5)} prompt ${String(r.prompt_eval_count).padStart(5)} tok  decode ${t.toFixed(1)} tok/s  ${(1000 / t).toFixed(2)} ms/token  prefill ${pref.toFixed(0)} tok/s  seq~${seq}`);
}
