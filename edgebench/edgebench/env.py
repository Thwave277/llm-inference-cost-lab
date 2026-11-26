"""edgebench.env — 环境快照。所有性能数字必须带这个快照，否则不可复现。"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from typing import Any, Dict

from .core import run_cmd, write_json

SMI_FIELDS = [
    "name", "memory.total", "driver_version", "clocks.max.sm", "clocks.max.mem",
    "clocks.current.sm", "power.limit", "power.draw", "pcie.link.gen.max",
    "pcie.link.width.max", "pcie.link.gen.current", "pcie.link.width.current",
    "temperature.gpu",
]


def _query(fields: list) -> dict:
    return run_cmd(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                    "--format=csv,noheader,nounits"], check=False)


def _nvidia_smi() -> Dict[str, Any]:
    """有些云卡/驱动不认某些字段（MIG、vGPU、老驱动），整条命令会直接失败。
    所以从全字段往下退，退到能出数为止。"""
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not found"}
    for fields in (SMI_FIELDS, SMI_FIELDS[:8], ["name", "memory.total", "driver_version"]):
        q = _query(fields)
        if q["rc"] != 0:
            continue
        gpus = []
        for line in q["out"].strip().splitlines():
            vals = [v.strip() for v in line.split(",")]
            row = dict(zip(fields, vals))
            for f in SMI_FIELDS:          # 缺的字段补空，下游不用做 None 判断
                row.setdefault(f, "")
            gpus.append(row)
        return {"available": True, "gpus": gpus, "fields": fields,
                "downgraded": fields != SMI_FIELDS}
    return {"available": False, "reason": "nvidia-smi 查询全部失败", "raw_error": q["err"][:300]}


def _pkg_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return ""


def _cmd_version(cmd: str) -> str:
    if not shutil.which(cmd):
        return ""
    r = run_cmd([cmd, "--version"], check=False, timeout=20)
    return (r["out"] + r["err"]).strip().splitlines()[0] if (r["out"] or r["err"]) else ""


def snapshot() -> Dict[str, Any]:
    try:
        import torch  # 延迟导入：没装 torch 的机器也能跑 env
    except Exception as e:
        torch = None
        _no_torch = "torch 未安装: %s" % e

    ver: Dict[str, Any] = {
        "torch": getattr(torch, "__version__", ""),
        "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", "") if torch else "",
        "cuda_available": bool(torch and torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else "",
        "cudnn": torch.backends.cudnn.version() if torch and torch.backends.cudnn.is_available() else None,
        "vllm": _pkg_version("vllm"),
        "sglang": _pkg_version("sglang"),
        "transformers": _pkg_version("transformers"),
        "llama_bench": _cmd_version("llama-bench"),
        "nvidia_smi": _cmd_version("nvidia-smi"),
    }
    env = {
        "when": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "gpu": _nvidia_smi(),
        "versions": ver,
    }
    if torch:
        try:
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                ver["vram_gb"] = round(p.total_memory / 1e9, 2)
                ver["sm"] = "%d.%d" % (p.major, p.minor)
        except Exception:
            pass
    else:
        ver["note"] = "torch 未安装"
    return env


def clock_hint(gpu_index: int = 0) -> str:
    """给租用机器用的锁频提示。云主机一般要 root，且部分平台不允许改。"""
    return (
        "想让测量稳定，先尝试锁频（需要 root，被拒就退回多次取中位数）：\n"
        "  nvidia-smi -i %d -lgc <max_sm_clock>      # 锁 SM 频率\n"
        "  nvidia-smi -i %d -lmc <max_mem_clock>     # 锁显存频率（部分卡不支持）\n"
        "  nvidia-smi -i %d -pl <watt>               # 锁功耗上限\n"
        "  nvidia-smi -rgc && nvidia-smi -rmc        # 恢复\n"
        "锁不了就用 --runs 5 以上取中位数，并把离散度一并记下来\n"
    ) % (gpu_index, gpu_index, gpu_index)


def main(out: str = "runs/env.json") -> Dict[str, Any]:
    env = snapshot()
    write_json(out, env)
    g = env["gpu"]
    line = "no gpu"
    if g.get("available") and g["gpus"]:
        g0 = g["gpus"][0]
        line = "%s %s MiB driver %s pcie %sx%s" % (
            g0["name"], g0["memory.total"], g0["driver_version"],
            g0.get("pcie.link.gen.max", "?"), g0.get("pcie.link.width.max", "?"))
    print("env -> %s\n  %s\n  torch %s cuda %s vllm %s" % (
        out, line, env["versions"]["torch"], env["versions"]["cuda_runtime"],
        env["versions"]["vllm"] or "-"))
    if g.get("available") and g["gpus"]:
        try:
            w = int(g["gpus"][0].get("pcie.link.width.current") or 16)
            if w < 8:
                print("  ! PCIe 实际宽度只有 x%d，模型越大越吃亏；能换 x16 槽就换" % w)
        except ValueError:
            pass
    return env


if __name__ == "__main__":
    main()
