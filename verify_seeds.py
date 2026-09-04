#!/usr/bin/env python3
"""Regression-check COMMISSION against a list of known-good seeds.

For each seed in the seeds file this runs

    <binary> --start <seed> --benchmark --output <tmpfile>

and calls the seed a PASS if the program reports it. --benchmark makes the run
terminate after PRINT_INTERVAL iterations; without it the program runs forever.

Usage
    ./verify_seeds.py                        # ./main-sb against ../seeds.txt
    ./verify_seeds.py -s my_seeds.txt
    ./verify_seeds.py -b ./main-lb -r 3      # three runs per seed
    ./verify_seeds.py --device 1

Exit status is the number of failing seeds, so it drops straight into CI.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# The binaries the makefile can produce, in the order we prefer them.
BINARY_NAMES = ("main-sb", "main-lb", "main-usb", "main-ulb", "main")

# A stats row looks like:
#   total   - 12833.960 ms | 99.9 % | 68719476736 ->   4 | ... | 5.352 Gips | ...
STATS_ROW = re.compile(
    r"^(?P<stage>\S+)\s+-\s+(?P<ms>[\d.]+)\s+ms\s*\|"
    r"[^|]*\|\s*(?P<in>\d+)\s*->\s*(?P<out>\d+)"
)


@dataclass
class Result:
    seed: str
    ok: bool
    detections: int = 0
    seconds: float = 0.0
    gpu_ms: float | None = None      # program's own 'total' row
    final_out: int | None = None     # candidates the GPU handed to the CPU
    note: str = ""
    positions: list[str] = field(default_factory=list)


def find_binary(explicit: str | None, repo: Path) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            sys.exit(f"binary not found: {p}")
        return p
    for name in BINARY_NAMES:
        p = repo / name
        if p.exists() and os.access(p, os.X_OK):
            return p
    sys.exit(
        f"no binary found in {repo} (looked for {', '.join(BINARY_NAMES)}).\n"
        "Build one first, e.g.  make"
    )


def read_seeds(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"seeds file not found: {path}")
    seeds: list[str] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            int(line)
        except ValueError:
            print(f"  warning: {path}:{lineno}: not a seed, skipping: {raw!r}",
                  file=sys.stderr)
            continue
        seeds.append(line)
    if not seeds:
        sys.exit(f"no seeds in {path}")
    return seeds


def parse_stats(stdout: str) -> tuple[float | None, int | None]:
    """Return (total ms, final stage output count) from the last stats table."""
    total_ms = final_out = None
    last_stage_out = None
    for line in stdout.splitlines():
        m = STATS_ROW.match(line.strip())
        if not m:
            continue
        if m.group("stage") == "total":
            total_ms = float(m.group("ms"))
            final_out = int(m.group("out"))
        else:
            last_stage_out = int(m.group("out"))
    # Older builds print no output count on the total row; fall back to the
    # last filter stage, which is what actually reaches the CPU.
    if final_out is None:
        final_out = last_stage_out
    return total_ms, final_out


def run_one(binary: Path, seed: str, workdir: Path, timeout: float,
            device: str | None, extra: list[str]) -> Result:
    out_file = workdir / f"{seed}.out"
    if out_file.exists():
        out_file.unlink()

    cmd = [str(binary), "--start", seed, "--benchmark", "--output", str(out_file)]
    if device:
        cmd += ["--device", device]
    cmd += extra

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=workdir, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return Result(seed, False, seconds=time.monotonic() - start,
                      note=f"timed out after {timeout:.0f}s")
    elapsed = time.monotonic() - start

    if proc.returncode != 0:
        head = (proc.stdout or "").strip().splitlines()
        return Result(seed, False, seconds=elapsed,
                      note=f"exit {proc.returncode}: {head[0][:60] if head else 'no output'}")

    # The seed is reported both on stdout and in the --output file; the file is
    # the authoritative record, so prefer it and fall back to stdout.
    positions: list[str] = []
    if out_file.exists():
        for line in out_file.read_text(errors="replace").splitlines():
            if line.startswith(seed + " "):
                positions.append(line.strip())
    if not positions:
        for line in proc.stdout.splitlines():
            if line.startswith(seed + " ") or line.startswith(seed + " at "):
                positions.append(line.strip())

    gpu_ms, final_out = parse_stats(proc.stdout)
    return Result(seed, bool(positions), len(positions), elapsed,
                  gpu_ms, final_out, "" if positions else "not reported",
                  positions)


def print_table(results: list[Result], repeats: int) -> None:
    width = max(len(r.seed) for r in results)
    width = max(width, len("SEED"))

    print()
    print("=" * (width + 54))
    print(f"{'SEED':<{width}}  {'RESULT':<6}  {'HITS':>4}  {'TIME':>7}  "
          f"{'GPU ms':>9}  {'CAND':>5}  NOTE")
    print("-" * (width + 54))
    for r in results:
        verdict = "PASS" if r.ok else "FAIL"
        gpu = f"{r.gpu_ms:9.1f}" if r.gpu_ms is not None else " " * 9
        cand = f"{r.final_out:5d}" if r.final_out is not None else " " * 5
        print(f"{r.seed:<{width}}  {verdict:<6}  {r.detections:>4}  "
              f"{r.seconds:6.1f}s  {gpu}  {cand}  {r.note}")
    print("-" * (width + 54))

    passed = sum(1 for r in results if r.ok)
    failed = len(results) - passed
    total_s = sum(r.seconds for r in results)
    suffix = f" ({repeats} runs each)" if repeats > 1 else ""
    print(f"{len(results)} seed{'s' if len(results) != 1 else ''}{suffix}"
          f" · {passed} passed · {failed} failed · {total_s:.1f}s total")
    if failed:
        print("\nFAILED: " + ", ".join(r.seed for r in results if not r.ok))
    print("=" * (width + 54))


def main() -> int:
    repo = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="Check that COMMISSION still finds a list of known seeds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit status equals the number of failing seeds.",
    )
    ap.add_argument("-s", "--seeds", default=None,
                    help="seed list (default: ./seeds.txt, else ../seeds.txt)")
    ap.add_argument("-b", "--binary", default=None,
                    help=f"binary to test (default: first of {', '.join(BINARY_NAMES)})")
    ap.add_argument("-r", "--repeat", type=int, default=1, metavar="N",
                    help="run each seed N times; the seed passes only if every run finds it")
    ap.add_argument("-t", "--timeout", type=float, default=900.0,
                    help="per-run timeout in seconds (default: 900)")
    ap.add_argument("--device", default=None, help="GPU index, passed through to --device")
    ap.add_argument("--keep-output", action="store_true",
                    help="keep the per-seed output files instead of using a temp dir")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="stop at the first failing seed")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="extra args after -- are passed to the binary")
    args = ap.parse_args()

    extra = [a for a in args.rest if a != "--"]

    if args.seeds:
        seeds_path = Path(args.seeds)
    else:
        seeds_path = repo / "seeds.txt"
        if not seeds_path.exists():
            seeds_path = repo.parent / "seeds.txt"

    binary = find_binary(args.binary, repo)
    seeds = read_seeds(seeds_path)

    if args.keep_output:
        workdir = repo / "verify_out"
        workdir.mkdir(exist_ok=True)
        cleanup = None
    else:
        # Run somewhere scratch: with no --output the program appends to
        # output.txt in the working directory, and we would rather not touch
        # the one in the repo.
        tmp = tempfile.mkdtemp(prefix="verify_seeds_")
        workdir = Path(tmp)
        cleanup = tmp

    print(f"binary : {binary}")
    print(f"seeds  : {seeds_path} ({len(seeds)} seeds)")
    print(f"workdir: {workdir}")
    if args.repeat > 1:
        print(f"repeat : {args.repeat} runs per seed")
    print()

    results: list[Result] = []
    try:
        for i, seed in enumerate(seeds, 1):
            # Runs are strictly serial: two of these at once would contend for
            # the GPU and make every timing meaningless.
            attempts = [run_one(binary, seed, workdir, args.timeout,
                                args.device, extra)
                        for _ in range(args.repeat)]
            best = attempts[0]
            if args.repeat > 1:
                ok = all(a.ok for a in attempts)
                best = Result(
                    seed, ok,
                    min(a.detections for a in attempts),
                    sum(a.seconds for a in attempts),
                    (sum(a.gpu_ms for a in attempts if a.gpu_ms is not None)
                     / max(1, sum(1 for a in attempts if a.gpu_ms is not None))
                     if any(a.gpu_ms is not None for a in attempts) else None),
                    attempts[0].final_out,
                    "" if ok else "failed in "
                                  f"{sum(1 for a in attempts if not a.ok)}/{args.repeat} runs",
                    attempts[0].positions,
                )
            results.append(best)

            mark = "PASS" if best.ok else "FAIL"
            # flush: each seed takes ~13s, and without this the progress lines
            # sit in the block buffer until exit whenever output is redirected.
            print(f"[{i}/{len(seeds)}] {mark}  {seed}  "
                  f"{best.detections} hit(s)  {best.seconds:.1f}s"
                  + (f"  {best.note}" if best.note else ""), flush=True)
            for p in best.positions[:3]:
                print(f"          {p}", flush=True)

            if args.stop_on_fail and not best.ok:
                print("\nstopping at first failure (--stop-on-fail)")
                break
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    finally:
        if results:
            print_table(results, args.repeat)
        if cleanup and not args.keep_output:
            shutil.rmtree(cleanup, ignore_errors=True)

    return sum(1 for r in results if not r.ok)


if __name__ == "__main__":
    sys.exit(main())
