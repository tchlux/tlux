# macos_utilization.py
# Python 3.x, stdlib only.

import ctypes as C
import time

# ----- Common CF/IOKit setup -----
cf = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
iokit = C.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
libc = C.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)

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

def cfnumber_to_float(n: CFTypeRef):
    # Try float64 first, else int64
    out_d = C.c_double()
    if cf.CFNumberGetValue(n, kCFNumberFloat64Type, C.byref(out_d)):
        return float(out_d.value)
    out_i = C.c_longlong()
    if cf.CFNumberGetValue(n, kCFNumberSInt64Type, C.byref(out_i)):
        return float(out_i.value)
    return None

def cfdict_to_python(d: CFDictionaryRef):
    if not d:
        return None
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
        if cf.CFGetTypeID(k.value) == cf.CFStringGetTypeID():
            key = cfstring_to_py(k.value)
        else:
            continue
        vt = cf.CFGetTypeID(v.value)
        if vt == cf.CFNumberGetTypeID():
            out[key] = cfnumber_to_float(v.value)
        elif vt == cf.CFDictionaryGetTypeID():
            out[key] = cfdict_to_python(v.value)
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

# ----- CPU utilization via Mach host_processor_info -----
mach_task_self = libc.mach_task_self
mach_task_self.restype = mach_port_t

mach_host_self = libc.mach_host_self
mach_host_self.restype = mach_port_t

host_processor_info = libc.host_processor_info
host_processor_info.argtypes = [
    mach_port_t, C.c_int,
    C.POINTER(C.c_uint),                      # outCPUCount
    C.POINTER(C.POINTER(C.c_uint)),           # outInfo array
    C.POINTER(C.c_uint)                       # outInfoCount
]
host_processor_info.restype = C.c_int

vm_deallocate = libc.vm_deallocate
vm_deallocate.argtypes = [mach_port_t, C.c_void_p, C.c_size_t]
vm_deallocate.restype = C.c_int

PROCESSOR_CPU_LOAD_INFO = 2
CPU_STATE_USER, CPU_STATE_SYSTEM, CPU_STATE_IDLE, CPU_STATE_NICE = 0,1,2,3
CPU_STATE_MAX = 4

def _read_cpu_ticks():
    host = mach_host_self()
    cpu_count = C.c_uint(0)
    info_cnt = C.c_uint(0)
    info = C.POINTER(C.c_uint)()
    kr = host_processor_info(host, PROCESSOR_CPU_LOAD_INFO,
                             C.byref(cpu_count),
                             C.byref(info),
                             C.byref(info_cnt))
    if kr != 0:
        raise OSError(kr, "host_processor_info failed")
    ncpu = cpu_count.value
    total = [0,0,0,0]
    try:
        # info is an array sized ncpu * CPU_STATE_MAX
        arr = C.cast(info, C.POINTER(C.c_uint * (ncpu * CPU_STATE_MAX))).contents
        for i in range(ncpu):
            base = i * CPU_STATE_MAX
            total[CPU_STATE_USER]  += arr[base + CPU_STATE_USER]
            total[CPU_STATE_SYSTEM]+= arr[base + CPU_STATE_SYSTEM]
            total[CPU_STATE_IDLE]  += arr[base + CPU_STATE_IDLE]
            total[CPU_STATE_NICE]  += arr[base + CPU_STATE_NICE]
    finally:
        vm_deallocate(mach_task_self(), info, info_cnt.value * C.sizeof(C.c_uint))
    return total

def cpu_util_percent(sample_interval=0.25):
    a = _read_cpu_ticks()
    time.sleep(sample_interval)
    b = _read_cpu_ticks()
    du = b[0]-a[0]
    ds = b[1]-a[1]
    di = b[2]-a[2]
    dn = b[3]-a[3]
    used = du + ds + dn
    tot = used + di
    return 100.0 * used / tot if tot else 0.0

# ----- GPU utilization via IORegistry PerformanceStatistics -----
def _ioreg_perfstats_for_service(classname: str):
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

def gpu_util_percent():
    # Common Apple GPU service names to try
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
            if k in d and isinstance(d[k], (int, float)):
                v = float(d[k])
                # Some keys may be 0..1 fraction; normalize if needed.
                return v*100.0 if v <= 1.0 else v
        # Some firmwares nest stats under a sub-dictionary
        for subk in ("GPU", "GLDriver", "PerformanceStatistics"):
            sub = d.get(subk)
            if isinstance(sub, dict):
                for k in ("GPU Busy", "Device Utilization %", "HW Busy", "Overall Utilization %"):
                    if k in sub:
                        v = float(sub[k])
                        return v*100.0 if v <= 1.0 else v
    return None

# ----- ANE utilization (rarely exposed unprivileged) -----
def ane_util_percent():
    for name in ("AppleNeuralEngine", "ANE", "ANEService"):
        d = _ioreg_perfstats_for_service(name)
        if not d:
            continue
        for k in ("ANE Busy", "Device Utilization %", "HW Busy", "Overall Utilization %"):
            if k in d:
                v = float(d[k])
                return v*100.0 if v <= 1.0 else v
    return None

def sample_all():
    return {
        "cpu_pct": cpu_util_percent(),
        "gpu_pct": gpu_util_percent(),
        "ane_pct": ane_util_percent(),
    }

if __name__ == "__main__":
    import time
    while True:
        print(sample_all())
        time.sleep(1)
