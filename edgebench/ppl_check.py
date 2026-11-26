#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""交叉校验：用 transformers 直接前向算同一批 chunk 的 PPL，和 vLLM 的 prompt_logprobs 对比。"""
import json
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/models/Qwen3-4B"
CORPUS = "/root/autodl-tmp/wikitext2-test.txt"
N_CHUNK, CHUNK = 2, 1024

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
model.eval()

text = open(CORPUS, encoding="utf-8", errors="ignore").read()
ids_all = tok.encode(text)
res = []
for k in range(N_CHUNK):
    ids = torch.tensor([ids_all[k * CHUNK:(k + 1) * CHUNK]], device="cuda")
    with torch.no_grad():
        out = model(ids, labels=ids)
    ppl = math.exp(out.loss.item())
    res.append(round(ppl, 4))
    print("chunk %d: loss %.4f ppl %.4f" % (k, out.loss.item(), ppl), flush=True)

print(json.dumps({"model": MODEL, "chunk_ppl": res}))
