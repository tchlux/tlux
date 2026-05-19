# macOS Resource Samples Report Zero RSS And CPU

Status: open
Severity: low

## Summary

During the long DRAMA embedding run, `resources` samples showed useful GPU
utilization but always reported `rss: 0` and `cpu_percent: 0.0`, even though the
embedding subprocess was actively running for over twelve minutes.

## Reproduction

Run the Fourth Wing one-document DRAMA build and inspect the tokenizer job's
resource file:

```bash
tail data/fourth_wing_hkm_index/.hkm_jobs/ids/<tokenizer_job>/resources
```

Observed samples:

```json
{"rss": 0, "cpu_percent": 0.0, "gpu_percent": 94.0}
```

The GPU value changed over time and showed high utilization, but RSS and CPU
remained zero.

## Expected

On macOS, resource samples should report nonzero RSS and a useful CPU estimate
for the active executor process when possible.

## Actual

The sampler's macOS fallback calls `ps`, but the recorded values stayed zero.
This makes the TUI and job logs much less useful for diagnosing memory pressure
or CPU bottlenecks during long local builds.

## Impact

Operational observability is incomplete. During the long-document run, the only
reliable health signal was GPU utilization and sparse stdout progress.

## Likely Fix Area

Review `monitor.py::proc_usage` and `jobs.py::worker` process collection on
macOS. Verify the sampled PID list includes the real model process and that
`ps -o rss= -p <pid>` and `ps -o %cpu= -p <pid>` are parsed correctly. Add a
macOS-safe unit or integration smoke test that asserts RSS is nonzero for a
known running process.
