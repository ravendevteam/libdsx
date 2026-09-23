from __future__ import annotations

import argparse
import gc
import io
import json
import os
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import tracemalloc
import zlib
from collections import deque
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import libdsx


DEFAULT_CASES = ("specification", "tiny_records", "large_ascii", "large_paragraph", "citations")
EXTRA_CASES = ("empty_deflate_blocks", "adjacent_text")
OPERATIONS = ("load", "validate_file", "dump", "render", "iter_records", "iter_render_file", "render_to_file", "dump_to_sink")


class Sink:
    def write(self, data) -> int:
        return len(data)


def document(case: str) -> libdsx.Document:
    metadata = libdsx.Metadata("BENCHMARK", ("Author",), 1, date(2026, 1, 1))
    if case == "tiny_records":
        records = tuple(libdsx.AsciiBlock("") for _ in range(20000))
    elif case == "large_ascii":
        records = (libdsx.AsciiBlock(("A" * 97 + "\n") * 10000),)
    elif case == "large_paragraph":
        records = (libdsx.Paragraph((libdsx.Text("word " * 150000 + "end"),)),)
    elif case == "citations":
        records = tuple(libdsx.Paragraph((libdsx.Citation(number),)) for number in range(1, 1001))
        records += tuple(libdsx.Reference(number, "https://example.com") for number in range(1, 1001))
    elif case == "adjacent_text":
        records = (libdsx.Paragraph(tuple(libdsx.Text("word ") for _ in range(20000)) + (libdsx.Text("end"),)),)
    elif case == "empty_deflate_blocks":
        records = ()
    else:
        raise ValueError(case)
    return libdsx.Document(metadata, records)


def encoded_case(case: str) -> bytes:
    if case == "specification":
        return libdsx.dumps(libdsx.load(ROOT / "tests" / "fixtures" / "DossierRev1.dsx"))
    encoded = libdsx.dumps(document(case))
    if case != "empty_deflate_blocks":
        return encoded
    content = b"\x78\x01" + b"\x00\x00\x00\xff\xff" * 100000 + b"\x01\x00\x00\xff\xff\x00\x00\x00\x01"
    header = bytearray(encoded[:32])
    metadata_length = struct.unpack_from("<I", header, 12)[0]
    struct.pack_into("<I", header, 20, len(content))
    struct.pack_into("<I", header, 28, zlib.crc32(header[:28]))
    return bytes(header) + encoded[32:32 + metadata_length] + content


def process_memory() -> tuple[int | None, int | None]:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = wintypes.HANDLE
        inspect = ctypes.windll.psapi.GetProcessMemoryInfo
        inspect.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        inspect.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not inspect(current_process(), ctypes.byref(counters), counters.cb):
            return None, None
        return counters.WorkingSetSize, counters.PeakWorkingSetSize
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak *= 1 if sys.platform == "darwin" else 1024
        current = None
        if sys.platform.startswith("linux"):
            current = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        return current, int(peak)
    except (ImportError, OSError, ValueError):
        return None, None


def operation_callback(case: str, name: str, encoded: bytes):
    value = None
    if name in ("dump", "render", "dump_to_sink"):
        value = document(case) if case == "adjacent_text" else libdsx.loads(encoded)
    if name == "load":
        return lambda: libdsx.load(io.BytesIO(encoded))
    if name == "validate_file":
        return lambda: libdsx.validate_file(io.BytesIO(encoded))
    if name == "dump":
        return lambda: libdsx.dump(value, io.BytesIO())
    if name == "render":
        return lambda: libdsx.render(value)
    if name == "iter_records":
        return lambda: deque(libdsx.iter_records(io.BytesIO(encoded)), maxlen=0)
    if name == "iter_render_file":
        return lambda: deque(libdsx.iter_render(io.BytesIO(encoded)), maxlen=0)
    if name == "render_to_file":
        return lambda: libdsx.render_to(io.BytesIO(encoded), Sink())
    if name == "dump_to_sink":
        return lambda: libdsx.dump(value, Sink())
    raise ValueError(name)


def worker(case: str, operation: str, source: Path, repeat: int) -> dict:
    encoded = source.read_bytes()
    callback = operation_callback(case, operation, encoded)
    gc.collect()
    rss_before, peak_before = process_memory()
    times = []
    for _ in range(repeat):
        gc.collect()
        started = time.perf_counter()
        result = callback()
        times.append(time.perf_counter() - started)
        del result
    rss_after, peak_after = process_memory()
    gc.collect()
    tracemalloc.start()
    result = callback()
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return {
        "seconds": statistics.median(times),
        "seconds_samples": times,
        "python_peak_bytes": python_peak,
        "process_rss_after_setup_bytes": rss_before,
        "process_rss_after_timing_bytes": rss_after,
        "process_peak_rss_after_setup_bytes": peak_before,
        "process_peak_rss_bytes": peak_after,
        "process_peak_rss_growth_bytes": None if peak_before is None or peak_after is None else max(0, peak_after - peak_before),
    }


def positive_integer(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DSX time, Python allocations, and isolated process memory.")
    parser.add_argument("--case", action="append", choices=DEFAULT_CASES + EXTRA_CASES)
    parser.add_argument("--operation", action="append", choices=OPERATIONS)
    parser.add_argument("--repeat", type=positive_integer, default=3)
    parser.add_argument("--include-pathological", action="store_true")
    parser.add_argument("--output", type=Path, help="Write machine-readable results to this JSON file.")
    parser.add_argument("--baseline", type=Path, help="Compare original operations with a saved baseline JSON file.")
    parser.add_argument("--worker", nargs=2, metavar=("CASE", "OPERATION"), help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.worker:
        if args.input is None:
            raise ValueError("worker requires --input")
        print(json.dumps(worker(*args.worker, args.input, args.repeat)))
        return 0
    cases = args.case or DEFAULT_CASES + (EXTRA_CASES if args.include_pathological else ())
    operations = args.operation or OPERATIONS
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else {}
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "repeat": args.repeat,
        "timing": "Median of untraced runs; each case/operation uses a fresh child process.",
        "python_allocations": "Peak traced allocation in a separate operation run, excluding workload setup.",
        "process_memory": "RSS/working set before and after untraced runs; process peak includes interpreter and setup; growth subtracts setup peak. Tracing is excluded.",
        "baseline_comparison": "Ratios compare elapsed time and Python allocation peaks only; the original baseline did not isolate process memory.",
        "streaming_operations": "iter_render_file and render_to_file read binary streams; rendered output is discarded. dump_to_sink discards compressed output. iter_records discards records.",
        "results": {},
    }
    print(f"{'Case':<22} {'Operation':<19} {'Median ms':>10} {'Python KiB':>12} {'Process MiB':>12} {'Time/base':>10} {'Peak/base':>10}", flush=True)
    with tempfile.TemporaryDirectory(prefix="libdsx-benchmark-") as temporary:
        for case in cases:
            encoded = encoded_case(case)
            source = Path(temporary) / f"{case}.dsx"
            source.write_bytes(encoded)
            result = {"stored_bytes": len(encoded), "operations": {}}
            report["results"][case] = result
            for operation in operations:
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", case, operation, "--input", str(source), "--repeat", str(args.repeat)]
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                if completed.returncode:
                    sys.stderr.write(completed.stderr)
                    return completed.returncode
                metrics = json.loads(completed.stdout)
                previous = baseline.get("results", {}).get(case, {}).get("operations", {}).get(operation)
                time_ratio = peak_ratio = "-"
                if previous:
                    metrics["baseline_ratios"] = {
                        "seconds": metrics["seconds"] / previous["seconds"],
                        "python_peak_bytes": metrics["python_peak_bytes"] / previous["python_peak_bytes"],
                    }
                    time_ratio = f"{metrics['baseline_ratios']['seconds']:.2f}x"
                    peak_ratio = f"{metrics['baseline_ratios']['python_peak_bytes']:.2f}x"
                result["operations"][operation] = metrics
                process_peak = "n/a" if metrics["process_peak_rss_bytes"] is None else f"{metrics['process_peak_rss_bytes'] / 1048576:.2f}"
                print(f"{case:<22} {operation:<19} {metrics['seconds'] * 1000:>10.3f} {metrics['python_peak_bytes'] / 1024:>12.2f} {process_peak:>12} {time_ratio:>10} {peak_ratio:>10}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"JSON: {args.output.resolve()}")
    print("Time excludes allocation tracing. Process peaks include interpreter/setup; each row runs in a separate process.")
    if args.baseline:
        print("Ratios below 1.00x indicate improvement; process memory has no original baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
