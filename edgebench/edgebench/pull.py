"""edgebench.pull — 下模型。国内机器优先 modelscope，其次 hf-mirror。"""
from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional

from .core import run_cmd


def _hf_cli() -> Optional[str]:
    for name in ("hf", "huggingface-cli"):
        if shutil.which(name):
            return name
    return None


def plan(repo: str, dest: str, source: str = "auto") -> List[Dict[str, Any]]:
    """按优先级给出候选方案，第一个跑通的就是要用的。

    每个方案是 {"cmd": [...], "env": {...}}；env 单独给，避免依赖 shell 的 "env VAR=x cmd" 写法。
    """
    plans: List[Dict[str, Any]] = []
    if source in ("auto", "modelscope") and shutil.which("modelscope"):
        plans.append({"cmd": ["modelscope", "download", "--model", repo, "--local_dir", dest],
                      "env": {}, "via": "modelscope"})

    cli = _hf_cli()
    if source in ("auto", "hf") and cli:
        base = [cli, "download", repo, "--local-dir", dest]
        if os.environ.get("HF_ENDPOINT"):
            plans.append({"cmd": base, "env": {}, "via": "hf"})
        else:
            plans.append({"cmd": base, "env": {}, "via": "hf"})
            plans.append({"cmd": base, "env": {"HF_ENDPOINT": "https://hf-mirror.com"},
                          "via": "hf-mirror"})
    return plans


def main(repo: str, dest: str, source: str = "auto", dry_run: bool = False) -> int:
    plans = plan(repo, dest, source)
    if not plans:
        print("没有可用的下载器。装一个：\n"
              "  pip install -U huggingface_hub\n  pip install -U modelscope")
        return 1

    print("目标目录: %s" % dest)
    for p in plans:
        print("  候选: [%s] %s" % (p["via"], " ".join(p["cmd"][:3])))
    if dry_run:
        return 0

    os.makedirs(dest, exist_ok=True)
    last: Optional[Dict[str, Any]] = None
    for p in plans:
        saved = {k: os.environ.get(k) for k in p["env"]}
        os.environ.update(p["env"])
        try:
            r = run_cmd(p["cmd"], check=False)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        if r["rc"] == 0:
            print("完成: %s" % dest)
            print("下一步: python -m edgebench sweep --model %s ..." % dest)
            return 0
        last = r
        print("  [%s] 失败 rc=%s，换下一个通道" % (p["via"], r["rc"]))

    print("全部通道失败。最后错误:\n%s" % ((last or {}).get("err", "")[-800:]))
    return 1
