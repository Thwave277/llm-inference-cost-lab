#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""包自检：一条命令确认「报告里写了的东西，代码库里真的有」。

    python edgebench/selftest.py

检查四件事：
  1. 所有 Python 脚本能编译（不落 __pycache__）
  2. 所有 JSON / JSONL 结果能解析
  3. 报告里用反引号引用的文件都存在
  4. costmodel 的自测能过（有 node 的话）
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEARCH = [ROOT, os.path.join(ROOT, "costmodel"), os.path.join(ROOT, "edgebench"),
          os.path.join(HERE, "runs-4090"), os.path.join(HERE, "runs")]

ok_n = fail_n = 0
fails = []


def check(name, cond, detail=""):
    global ok_n, fail_n
    if cond:
        ok_n += 1
    else:
        fail_n += 1
        fails.append("%s %s" % (name, detail))


# 1. 脚本可编译
py_files = []
for d in (HERE, os.path.join(HERE, "edgebench")):
    if os.path.isdir(d):
        py_files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".py")]
for p in py_files:
    try:
        compile(open(p, encoding="utf-8").read(), p, "exec")
        check("compile", True)
    except SyntaxError as e:
        check("compile", False, "%s: %s" % (os.path.basename(p), e))

# 2. 数据可解析
data_dir = os.path.join(HERE, "runs-4090")
n_json = n_jsonl = 0
if os.path.isdir(data_dir):
    for f in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, f)
        try:
            if f.endswith(".json"):
                json.load(open(p, encoding="utf-8"))
                n_json += 1
            elif f.endswith(".jsonl"):
                for line in open(p, encoding="utf-8"):
                    line = line.strip()
                    if line:
                        json.loads(line)
                n_jsonl += 1
            else:
                continue
            check("json", True)
        except Exception as e:
            check("json", False, "%s: %s" % (f, e))
check("data-count", n_json + n_jsonl >= 25, "只找到 %d 个结果文件" % (n_json + n_jsonl))

# 3. 报告引用的文件
rep_path = os.path.join(ROOT, "costmodel", "REPORT.md")
refs = set()
if os.path.exists(rep_path):
    text = open(rep_path, encoding="utf-8").read()
    refs = set(re.findall(r"`([A-Za-z0-9_\-./]+\.(?:py|mjs|json|jsonl|md|sh|txt|yml|yaml))`", text))
for r in sorted(refs):
    found = any(os.path.exists(os.path.join(s, r)) or
                os.path.exists(os.path.join(s, os.path.basename(r))) for s in SEARCH)
    check("ref", found, r)

# 4. costmodel 自测（可选）
node = shutil_which = None
for cand in ("node", r"D:\New Folder\node.exe"):
    try:
        out = subprocess.run([cand, "--version"], capture_output=True, timeout=10)
        if out.returncode == 0:
            node = cand
            break
    except Exception:
        pass
if node:
    cm = os.path.join(ROOT, "costmodel", "costmodel.mjs")
    if os.path.exists(cm):
        r = subprocess.run([node, cm, "--selftest"], capture_output=True, timeout=60)
        out = (r.stdout or b"").decode("utf-8", "replace") + (r.stderr or b"").decode("utf-8", "replace")
        check("costmodel-selftest", "6/6 passed" in out, out.strip()[-60:])

print("脚本 %d 个 | 数据 %d 个 json + %d 个 jsonl | 报告引用 %d 项" % (len(py_files), n_json, n_jsonl, len(refs)))
print("通过 %d，失败 %d" % (ok_n, fail_n))
if fails:
    for f in fails:
        print("  FAIL", f)
    sys.exit(1)
print("包自检通过：报告里引用的脚本与数据都在，JSON 全部可解析")
