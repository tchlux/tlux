# GPU utilization sampler for macOS.
# 
# One-shot helper that reads the IORegistry PerformanceStatistics entry
# exposed by Apple GPU services and returns the current utilization
# percentage. Designed to be dependency-free and suitable for lightweight
# polling on Apple Silicon hardware.
# 
# Example:
#     pct = gpu_util_percent()
#     print(f"GPU utilization: {pct:.1f}%")


import ctypes as C

# Common CF/IOKit setup
cf = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
iokit = C.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")

CFTypeRef = C.c_void_p
CFStringRef = C.c_void_p
CFDictionaryRef = C.c_void_p
CFAllocatorRef = C.c_void_p
CFIndex = C.c_long
Boolean = C.c_ubyte
UInt32 = C.c_uint32
mach_port_t = C.c_uint

kCFStringEncodingUTF8 = 0x08000100

# CF prototypes
cf.CFStringCreateWithCString.argtypes = [CFAllocatorRef, C.c_char_p, C.c_uint32]
cf.CFStringCreateWithCString.restype = CFStringRef
cf.CFStringGetTypeID.restype = C.c_ulong
cf.CFNumberGetTypeID.restype = C.c_ulong
cf.CFDictionaryGetTypeID.restype = C.c_ulong
cf.CFGetTypeID.argtypes = [CFTypeRef]
cf.CFGetTypeID.restype = C.c_ulong
cf.CFRelease.argtypes = [CFTypeRef]
cf.CFRelease.restype = None

cf.CFStringGetCString.argtypes = [CFStringRef, C.c_char_p, C.c_long, C.c_uint32]
cf.CFStringGetCString.restype = Boolean

cf.CFNumberGetValue.argtypes = [CFTypeRef, C.c_int, C.c_void_p]
cf.CFNumberGetValue.restype = Boolean
kCFNumberSInt64Type = 4
kCFNumberFloat64Type = 6

cf.CFDictionaryGetCount.argtypes = [CFDictionaryRef]
cf.CFDictionaryGetCount.restype = CFIndex
cf.CFDictionaryGetKeysAndValues.argtypes = [
    CFDictionaryRef,
    C.POINTER(C.c_void_p),
    C.POINTER(C.c_void_p),
]
cf.CFDictionaryGetKeysAndValues.restype = None

def CFSTR(py_str: str) -> CFStringRef:
    return cf.CFStringCreateWithCString(None, py_str.encode("utf-8"), kCFStringEncodingUTF8)

def cfstring_to_py(s: CFStringRef) -> str:
    buf = C.create_string_buffer(1024)
    if cf.CFStringGetCString(s, buf, len(buf), kCFStringEncodingUTF8):
        return buf.value.decode("utf-8", "replace")
    # Fallback: try larger
    n = 4096
    buf = C.create_string_buffer(n)
    if cf.CFStringGetCString(s, buf, n, kCFStringEncodingUTF8):
        return buf.value.decode("utf-8", "replace")
    return ""

def cfnumber_to_float(n: CFTypeRef) -> float | None:
    # Try float64 first, else int64
    out_d = C.c_double()
    if cf.CFNumberGetValue(n, kCFNumberFloat64Type, C.byref(out_d)):
        return float(out_d.value)
    out_i = C.c_longlong()
    if cf.CFNumberGetValue(n, kCFNumberSInt64Type, C.byref(out_i)):
        return float(out_i.value)
    return None

def cfdict_to_python(d: CFDictionaryRef) -> dict[str, float | dict] | None:
    if cf.CFGetTypeID(d) != cf.CFDictionaryGetTypeID():
        return None
    count = cf.CFDictionaryGetCount(d)
    keys = (C.c_void_p * count)()
    vals = (C.c_void_p * count)()
    cf.CFDictionaryGetKeysAndValues(d, keys, vals)
    out = {}
    for i in range(count):
        k = C.c_void_p(keys[i])
        v = C.c_void_p(vals[i])
        if (cf.CFGetTypeID(k.value) == cf.CFStringGetTypeID()) and (k.value is not None):
            key = cfstring_to_py(C.cast(k.value, CFStringRef))
        else:
            continue
        vt = cf.CFGetTypeID(v.value)
        if (vt == cf.CFNumberGetTypeID()) and (v.value is not None):
            out[key] = cfnumber_to_float(C.cast(v.value, CFStringRef))
        elif (vt == cf.CFDictionaryGetTypeID()) and (v.value is not None):
            out[key] = cfdict_to_python(C.cast(v.value, CFStringRef))
        else:
            # unsupported CFType -> skip
            continue
    return out

# IOKit prototypes
iokit.IOServiceMatching.argtypes = [C.c_char_p]
iokit.IOServiceMatching.restype = CFDictionaryRef

iokit.IOServiceGetMatchingService.argtypes = [mach_port_t, CFDictionaryRef]
iokit.IOServiceGetMatchingService.restype = C.c_uint  # io_service_t

iokit.IORegistryEntryCreateCFProperty.argtypes = [
    C.c_uint,         # io_registry_entry_t
    CFStringRef,      # key
    CFAllocatorRef,   # allocator
    C.c_uint          # options
]
iokit.IORegistryEntryCreateCFProperty.restype = CFTypeRef

iokit.IOObjectRelease.argtypes = [C.c_uint]
iokit.IOObjectRelease.restype = C.c_int

# GPU utilization via IORegistry PerformanceStatistics
def _ioreg_perfstats_for_service(classname: str) -> dict[str, float | dict] | None:
    match = iokit.IOServiceMatching(classname.encode("utf-8"))
    if not match:
        return None
    svc = iokit.IOServiceGetMatchingService(0, match)
    if not svc:
        return None
    try:
        key = CFSTR("PerformanceStatistics")
        try:
            cfobj = iokit.IORegistryEntryCreateCFProperty(svc, key, None, 0)
        finally:
            cf.CFRelease(key)
        if not cfobj:
            return None
        try:
            return cfdict_to_python(cfobj)
        finally:
            cf.CFRelease(cfobj)
    finally:
        iokit.IOObjectRelease(svc)

# Description:
#   Return the current GPU utilization percentage from IORegistry, or None
#   if no supported GPU service exposes PerformanceStatistics.
#
# Returns:
#   float or None: Utilization in [0, 100] when available.
#
# Example:
#   pct = gpu_util_percent()
#   print(f"GPU utilization: {pct:.1f}%")
def gpu_util_percent() -> float | None:
    for name in ("AGXAccelerator", "IOAccelerator", "AppleGPUWrangler"):
        d = _ioreg_perfstats_for_service(name)
        if not d:
            continue
        # Known keys vary by OS/SoC. Check several candidates.
        for k in (
            "GPU Busy",
            "Device Utilization %",
            "Renderer Utilization %",
            "HW Busy",
            "Overall Utilization %",
        ):
            if (k in d):
                dk = d[k]
                if (isinstance(dk, (int, float))):
                    v = float(dk)
                    return v * 100.0 if v <= 1.0 else v
        # Some firmwares nest stats under a sub-dictionary.
        for subk in ("GPU", "GLDriver", "PerformanceStatistics"):
            sub = d.get(subk)
            if isinstance(sub, dict):
                for k in (
                    "GPU Busy",
                    "Device Utilization %",
                    "HW Busy",
                    "Overall Utilization %",
                ):
                    if k in sub and isinstance(sub[k], (int, float)):
                        v = float(sub[k])
                        return v * 100.0 if v <= 1.0 else v
    return None


import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

prev_cpu: Optional[Tuple[float, float]] = None  # (proc_time, wall_time)

# Return {'rss','cpu_percent','gpu_percent'} for the hottest PID.
def proc_usage(pids: list[int]) -> Dict[str, Any]:
    best = {"rss": 0, "cpu_percent": 0.0, "gpu_percent": None}
    for pid in pids:
        rss = 0
        cpu_pct = 0.0
        stat_path = Path(f"/proc/{pid}/stat")
        status_path = Path(f"/proc/{pid}/status")
        if stat_path.exists() and status_path.exists():
            with stat_path.open() as f:
                parts = f.readline().split()
                ut, st = int(parts[13]), int(parts[14])
            clk = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
            proc_time = (ut + st) / clk
            wall = time.time()
            global prev_cpu
            if prev_cpu is not None:
                dt_cpu = proc_time - prev_cpu[0]
                dt_wall = wall - prev_cpu[1]
                if dt_wall > 0:
                    cpu_pct = 100 * dt_cpu / dt_wall
            prev_cpu = (proc_time, wall)
            with status_path.open() as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
                        break
        else:
            try:
                out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
                rss_kb = int(out.strip() or 0)
                rss = rss_kb * 1024
            except Exception:
                rss = 0
            try:
                out = subprocess.check_output(["ps", "-o", "%cpu=", "-p", str(pid)], text=True)
                cpu_pct = float(out.strip() or 0.0)
            except Exception:
                cpu_pct = 0.0
        gpu_pct = gpu_util_percent()
        if rss > best["rss"]:
            best["rss"] = rss
        if cpu_pct > best["cpu_percent"]:
            best["cpu_percent"] = cpu_pct
        if gpu_pct is not None:
            if best["gpu_percent"] is None or gpu_pct > best["gpu_percent"]:
                best["gpu_percent"] = gpu_pct
    return best


if __name__ == "__main__":
    while True:
        print()
        usage = proc_usage([os.getpid()])
        mem = usage["rss"]
        print(f"MEM {mem/2**20:.0f}MB" if mem is not None else "MEM utilization unavailable")
        pct = usage["cpu_percent"]
        print(f"CPU {pct:.1f}%" if pct is not None else "CPU utilization unavailable")
        pct = usage["gpu_percent"]
        print(f"GPU {pct:.1f}%" if pct is not None else "GPU utilization unavailable")
        time.sleep(1)
