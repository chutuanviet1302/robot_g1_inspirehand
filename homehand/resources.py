"""Resource guard: never start more simulation workers than the free RAM can hold (Linux and Windows)."""

from __future__ import annotations

import os

WORKER_MB = 900        # measured peak RSS of one simulation worker with the head-camera renderer
WORKER_TORCH_MB = 2200  # measured: a worker that also loads PyTorch + YOLO / a learned policy (~2.1 GB RSS)
RESERVE_MB = 2500      # left for the OS, the browser and the web server


def available_mb() -> int:
    try:
        import psutil  # optional
        return int(psutil.virtual_memory().available / 2**20)
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes

        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MemStatus()
        st.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return int(st.ullAvailPhys / 2**20)
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return 4096


def safe_workers(requested: int | None = None, per_worker_mb: int = WORKER_MB) -> int:
    """Number of worker processes that fits in free RAM (and leaves two CPU threads free)."""
    by_cpu = max(1, (os.cpu_count() or 2) - 2)
    by_ram = max(1, (available_mb() - RESERVE_MB) // per_worker_mb)
    n = min(by_cpu, by_ram, 6)
    if requested is not None:
        n = min(requested, n)
    return max(1, int(n))
