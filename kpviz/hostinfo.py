"""What this machine will actually give us.

`os.cpu_count()` and `SC_PHYS_PAGES` report the *host*, which is wrong under
CPU affinity masks (taskset / Slurm cpusets / numactl) and wrong under cgroup
quotas (Docker / Kubernetes / HPC). Getting these wrong is not a small
mis-tuning: telling DuckDB it may use 256 GB inside a 4 GB container ends the
scan with the OOM killer, and spawning 64 workers on a 4-core cpuset makes
everything slower than 4 workers would have been.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def usable_cpus() -> int:
    """CPUs this process may actually run on, honouring affinity + cgroups."""
    n = None
    if hasattr(os, "process_cpu_count"):        # Python 3.13+: honours affinity
        try:
            n = os.process_cpu_count()
        except Exception:
            n = None
    if not n and hasattr(os, "sched_getaffinity"):
        try:
            n = len(os.sched_getaffinity(0))
        except Exception:
            n = None
    n = n or os.cpu_count() or 2

    # cgroup v2 quota
    try:
        txt = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if txt and txt[0] != "max":
            quota, period = int(txt[0]), int(txt[1])
            if period > 0:
                n = min(n, max(1, round(quota / period)))
    except Exception:
        pass
    # cgroup v1 quota
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0 and p > 0:
            n = min(n, max(1, round(q / p)))
    except Exception:
        pass
    return max(1, int(n))


@lru_cache(maxsize=1)
def usable_ram_bytes() -> int:
    """RAM this process may actually use, honouring cgroup limits."""
    total = None
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        total = None
    if not total:
        total = 8 * 2**30
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            txt = Path(p).read_text().strip()
            if txt and txt != "max":
                v = int(txt)
                # cgroup v1 uses a huge sentinel for "unlimited"
                if 0 < v < (1 << 62):
                    total = min(total, v)
        except Exception:
            continue
    return int(total)


def describe() -> dict:
    return {"usable_cpus": usable_cpus(),
            "host_cpus": os.cpu_count(),
            "usable_ram_gb": round(usable_ram_bytes() / 2**30, 1)}
