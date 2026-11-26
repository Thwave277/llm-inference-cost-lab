"""edgebench.precision — 精度回归门槛。

两类检查：
  1. perplexity：同一份语料、同一个模型，比较两次构建（或两套开关）的 PPL 差异
  2. 任务集：一组带标准答案的短题（数学 / 代码 / 长逻辑各几条），跑过率对比

用法约定：任何"提速"实验，只要 PPL 涨超过 tol、或任务集通过率掉超过 tol，就不算通过。
默认容忍：PPL 变化 <= 2%，通过率下降 <= 1 题。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from .core import run_cmd


def _exe(bin_dir: str, name: str) -> str:
    p = os.path.join(bin_dir, name)
    return p + ".exe" if os.name == "nt" and not p.endswith(".exe") else p


# --------------------------------------------------------------------------- #
# perplexity
# --------------------------------------------------------------------------- #
PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)")
PPL_LOOSE = re.compile(r"(?<![A-Za-z.])PPL\s*=\s*([0-9.]+)")


def _extract_ppl(text: str) -> Optional[float]:
    m = PPL_RE.search(text)
    if m:
        return float(m.group(1))
    hits = PPL_LOOSE.findall(text)      # 老版本直接打 "PPL = x"，取最后一个（最终值）
    return float(hits[-1]) if hits else None


def perplexity(bin_dir: str, model_path: str, corpus: str, ctx: int = 512,
               ngl: int = 99, extra_args: Optional[Sequence[str]] = None,
               timeout: float = 3600) -> Dict[str, Any]:
    cmd = [_exe(bin_dir, "llama-perplexity"), "-m", model_path, "-f", corpus,
           "-c", str(ctx), "-ngl", str(ngl)]
    if extra_args:
        cmd += list(extra_args)
    r = run_cmd(cmd, timeout=timeout, check=False)
    ppl = _extract_ppl(r["out"] + r["err"])
    return {
        "ok": ppl is not None, "ppl": ppl,
        "cmd": " ".join(cmd), "err": r["err"][-400:] if ppl is None else "",
    }


# --------------------------------------------------------------------------- #
# 任务集
# --------------------------------------------------------------------------- #
def load_tasks(path: str) -> List[Dict[str, Any]]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


NEG_CHARS = "不非没未"


def _contains_word(norm: str, token: str) -> bool:
    """单字判断（是/否/对/错）防否定。

    看回答的开头 8 个字：以否定词开头直接判否；单字紧跟在否定词后面也判否。
    不去全文找，否则"都是""都是C"这类叙述会把否定回答也判成肯定。
    """
    head = norm[:8]
    if not head:
        return False
    if head[0] in NEG_CHARS:
        return False
    i = head.find(token)
    if i < 0:
        return False
    return not (i > 0 and head[i - 1] in NEG_CHARS)


def _matches(text: str, expects: Sequence[str]) -> bool:
    """按期望值类型分别判断：

    * 纯数字（含小数/负数）：要求**独立出现**，避免 "391" 命中 "3910"、期望 "3" 命中 "13"
    * 单字（是/否/对/错）：排除被否定的情况
    * 其他（单词/短语）：忽略大小写与空白的子串匹配
    """
    if not text:
        return False
    norm = re.sub(r"\s+", "", text).lower()
    for e in expects:
        en = re.sub(r"\s+", "", str(e)).lower()
        if not en:
            continue
        if re.fullmatch(r"-?\d+(\.\d+)?", en):
            if not re.search(r"(?<![\d.])" + re.escape(en) + r"(?![\d.])", norm):
                return False
        elif len(en) == 1:
            if not _contains_word(norm, en):
                return False
        elif en not in norm:
            return False
    return True


def run_tasks_llamacpp(bin_dir: str, model_path: str, tasks: List[Dict[str, Any]],
                       ngl: int = 99, max_tokens: int = 192,
                       extra_args: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    results = []
    for t in tasks:
        cmd = [_exe(bin_dir, "llama-cli"), "-m", model_path, "-p", t["prompt"],
               "-n", str(max_tokens), "-ngl", str(ngl), "-no-cnv", "--temp", "0"]
        if extra_args:
            cmd += list(extra_args)
        r = run_cmd(cmd, timeout=600, check=False)
        out = r["out"]
        ok = r["rc"] == 0 and _matches(out, t.get("expect", []))
        results.append({"prompt": t["prompt"][:60], "ok": ok, "rc": r["rc"]})
    passed = sum(1 for x in results if x["ok"])
    return {"engine": "llamacpp", "passed": passed, "total": len(results),
            "rate": round(passed / max(1, len(results)), 3), "detail": results}


def run_tasks_vllm(model: str, tasks: List[Dict[str, Any]], max_tokens: int = 192,
                   **llm_kwargs: Any) -> Dict[str, Any]:
    from vllm import LLM, SamplingParams

    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    outs = llm.generate([t["prompt"] for t in tasks], sp, use_tqdm=False)
    results = []
    for t, o in zip(tasks, outs):
        text = o.outputs[0].text if o.outputs else ""
        results.append({"prompt": t["prompt"][:60], "ok": _matches(text, t.get("expect", [])),
                        "text": text[:200]})
    passed = sum(1 for x in results if x["ok"])
    return {"engine": "vllm", "passed": passed, "total": len(results),
            "rate": round(passed / max(1, len(results)), 3), "detail": results}


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #
def judge(pk: Optional[Dict[str, Any]], pt: Optional[Dict[str, Any]],
          loss: Optional[Dict[str, Any]], task: Optional[Dict[str, Any]],
          ppl_tol: float = 0.02, task_tol: int = 1) -> Dict[str, Any]:
    """pk/pt: baseline / treatment 的 perplexity 结果；loss/task: 任务集结果。"""
    checks = []
    if pk and pt and pk.get("ppl") and pt.get("ppl"):
        delta = (pt["ppl"] - pk["ppl"]) / pk["ppl"]
        checks.append({"what": "ppl", "base": pk["ppl"], "new": pt["ppl"],
                       "delta": round(delta, 4), "pass": delta <= ppl_tol})
    if loss and task:
        drop = loss["passed"] - task["passed"]
        checks.append({"what": "tasks", "base": loss["passed"], "new": task["passed"],
                       "drop": drop, "pass": drop <= task_tol})
    ok = all(c["pass"] for c in checks) if checks else False
    return {"pass": ok, "checks": checks}
