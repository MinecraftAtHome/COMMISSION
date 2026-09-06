#!/usr/bin/env python3
"""Benchmark a range of commits and draw the result as a line chart.

Builds every commit from HEAD~N to HEAD, times each one with --benchmark, and
writes an SVG/HTML chart of milliseconds per search iteration.

    ./bench_history.py --from a1b2c3d         # that commit through HEAD
    ./bench_history.py --from v1.2 --verify   # any revision git understands
    ./bench_history.py -n 13                  # or count back: HEAD~13 .. HEAD
    ./bench_history.py -n 5 --runs 6          # more samples per commit
    ./bench_history.py -n 5 --arch sm_75      # pass ARCH= through to make

--from takes anything git resolves -- a hash of any length, a tag, a branch, or
HEAD~7 -- and works out the range itself. If the commit is not an ancestor of
HEAD the two histories have diverged, so it falls back to their merge base and
says so, since walking "backwards" to a sibling branch would not make a sensible
progress chart.

Nothing outside the output directory is touched: the commits are built in a
temporary git worktree, so the working tree, the index and the current branch
are all left exactly as they were, even on Ctrl-C.

Commits are built in parallel, each worker in its own git worktree, so a long
range does not mean a long wait. Worktrees share the object store, so N of them
cost N working trees on disk rather than N clones. --build-workers sets the
count; each one runs a full nvcc, which is memory-hungry, so the default is
deliberately modest.

Only the standard library is used, so the chart renders anywhere Python does --
no matplotlib, no numpy.

Two things about the measurement are worth knowing, because both were learned
the hard way on this project:

  * Commits are timed round-robin, not one after another. A GPU that starts cold
    and heats up makes whichever binary ran first look fastest; measured
    sequentially, an optimisation here once appeared to be a regression. Every
    commit is built first, then one run of each is taken per round.

  * The first rounds are discarded, and enough of them to reach thermal steady
    state. This matters more than it sounds: on the card this was written for,
    the same binary reads 26.4 ms/iteration cold and 30.2 ms/iteration hot, a
    15% spread, as the clock throttles from 1328 MHz at 40C to 1164 MHz at 79C.
    Warm-up is therefore counted in runs, not rounds, so a two-commit range
    warms up as thoroughly as a twenty-commit one.

Even so, two identical binaries can differ by around 1% run to run, so the
chart prints a standard error beside each point. Treat differences smaller than
a couple of standard errors as noise and raise --runs if you need to resolve
them.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import queue
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# The stats table the program prints at the end of a --benchmark run:
#   total   - 12833.960 ms | 99.9 % | 68719476736 ->   4 | ...
STATS_ROW = re.compile(
    r"^(?P<stage>\S+)\s+-\s+(?P<ms>[\d.]+)\s+ms\s*\|"
    r"[^|]*\|\s*(?P<in>\d+)\s*->\s*(?P<out>\d+)"
)

BINARY_GLOBS = ("main-sb", "main-lb", "main-usb", "main-ulb", "main")

# A neutral, colourblind-safe pair: teal for the timing line, rust for the
# correctness track. Both sit far enough apart under deuteranopia and protanopia
# to stay distinguishable, and each is paired with a direct label anyway.
LIGHT = dict(bg="#F6F7F9", card="#FFFFFF", ink="#14181F", ink2="#39424E",
             muted="#5B6572", faint="#8A94A1", line="#DCE1E7", grid="#EAEEF2",
             accent="#0E8FA3", warn="#B4521E")
DARK = dict(bg="#0F1318", card="#161B22", ink="#E4E9F0", ink2="#C3CBD6",
            muted="#8B95A3", faint="#6C7683", line="#252C36", grid="#1F262F",
            accent="#1AA3B8", warn="#CE7A3E")


@dataclass
class Commit:
    sha: str
    short: str
    subject: str
    binary: Path | None = None
    build_log: str = ""
    runs: list[float] = field(default_factory=list)   # ms per iteration
    seeds_scanned: int = 0
    final_out: int = 0
    verified: int | None = None                       # seeds found, if --verify
    verify_total: int | None = None

    @property
    def ok(self) -> bool:
        return bool(self.runs)

    @property
    def mean(self) -> float:
        return statistics.mean(self.runs) if self.runs else math.nan

    @property
    def stderr(self) -> float:
        # One sample has an unknown error, not a zero one; callers show a dash.
        if len(self.runs) < 2:
            return 0.0
        return statistics.stdev(self.runs) / math.sqrt(len(self.runs))


def run(cmd: list[str], cwd: Path | None = None, timeout: float | None = None,
        env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, timeout=timeout, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace")


def git(repo: Path, *args: str) -> str:
    r = run(["git", "-C", str(repo), *args])
    if r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{r.stdout}")
    return r.stdout.strip()


def _rev_list(repo: Path, spec: list[str], first_parent: bool) -> list[str]:
    args = ["rev-list", "--reverse"] + (["--first-parent"] if first_parent else [])
    return git(repo, *args, *spec).split()


def resolve_commits(repo: Path, n: int | None, from_commit: str | None,
                    first_parent: bool, max_commits: int) -> list[Commit]:
    """The commits to chart, oldest first.

    Either the last N+1 commits, or everything from `from_commit` through HEAD.
    """
    head = git(repo, "rev-parse", "HEAD")

    if from_commit:
        probe = run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet",
                     f"{from_commit}^{{commit}}"])
        if probe.returncode != 0 or not probe.stdout.strip():
            sys.exit(f"cannot resolve {from_commit!r} to a commit in {repo}")
        start = probe.stdout.strip()

        if start == head:
            sys.exit("--from is HEAD itself; there is nothing to compare it against")

        ancestor = run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                        start, "HEAD"]).returncode == 0
        if not ancestor:
            base = git(repo, "merge-base", start, "HEAD")
            ahead = len(_rev_list(repo, [f"{base}..{start}"], False))
            short_s = git(repo, "rev-parse", "--short", start)
            short_b = git(repo, "rev-parse", "--short", base)
            print(f"  note: {short_s} is not an ancestor of HEAD -- the histories "
                  f"diverged {ahead} commit(s) ago.")
            print(f"        charting from their merge base {short_b} instead.")
            start = base
            if start == head:
                sys.exit("the merge base is HEAD; nothing to chart")

        shas = [start] + _rev_list(repo, [f"{start}..HEAD"], first_parent)
    else:
        shas = _rev_list(repo, [f"--max-count={n + 1}", "HEAD"], first_parent)

    if len(shas) < 2:
        sys.exit(f"need at least two commits to chart; found {len(shas)}")
    if n is not None and len(shas) < n + 1:
        print(f"  note: history has only {len(shas)} commits, charting all of them")
    if len(shas) > max_commits:
        mins = len(shas) * 1.5
        sys.exit(f"that range is {len(shas)} commits, over the --max-commits limit "
                 f"of {max_commits}.\n"
                 f"Each one is built and timed, so expect upwards of {mins:.0f} "
                 f"minutes.\nRaise --max-commits if that is what you want.")

    commits = []
    for sha in shas:
        commits.append(Commit(sha=sha,
                              short=git(repo, "rev-parse", "--short", sha),
                              subject=git(repo, "log", "-1", "--format=%s", sha)))
    return commits


def build_commit(wt: Path, c: Commit, stage: Path, make_args: list[str],
                 print_interval: int, jobs: int) -> bool:
    """Check the commit out in the worktree and build it into `stage`."""
    r = run(["git", "-C", str(wt), "checkout", "--detach", "--force", c.sha])
    if r.returncode != 0:
        c.build_log = r.stdout
        return False

    # Wipe every build product before each commit. Two reasons: make does not
    # notice that -DPRINT_INTERVAL changed, so it would happily keep a stale
    # object; and some early commits have object files committed to the tree,
    # which would be linked instead of rebuilt. `git clean -xfd` removes the
    # untracked ones, and the explicit unlink handles the tracked ones.
    run(["git", "-C", str(wt), "clean", "-xfdq"])
    for pat in ("*.o", "*.a", "main", "main-*"):
        for p in wt.glob(pat):
            if p.is_file():
                p.unlink()

    cmd = ["make", f"-j{jobs}", f"PRINT_INTERVAL={print_interval}", *make_args]
    r = run(cmd, cwd=wt, timeout=3600)
    c.build_log = r.stdout
    if r.returncode != 0:
        return False

    for name in BINARY_GLOBS:
        p = wt / name
        if p.is_file() and os.access(p, os.X_OK):
            dst = stage / f"{c.short}-{name}"
            shutil.copy2(p, dst)
            c.binary = dst
            return True
    c.build_log += "\n(build reported success but produced no known binary)"
    return False


def time_once(c: Commit, seed: str, print_interval: int, workdir: Path,
              timeout: float, extra: list[str]) -> bool:
    """One --benchmark run. Appends ms/iteration to the commit's samples."""
    out_file = workdir / f"{c.short}.out"
    cmd = [str(c.binary), "--start", seed, "--benchmark",
           "--output", str(out_file), *extra]
    try:
        r = run(cmd, cwd=workdir, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    if r.returncode != 0:
        return False

    total_ms = seeds = final = None
    for line in r.stdout.splitlines():
        m = STATS_ROW.match(line.strip())
        if m and m.group("stage") == "total":
            total_ms = float(m.group("ms"))
            seeds = int(m.group("in"))
            final = int(m.group("out"))
    if total_ms is None:
        return False

    c.runs.append(total_ms / print_interval)
    c.seeds_scanned = seeds or 0
    c.final_out = final or 0
    return True


def verify_commit(repo: Path, c: Commit, seeds_file: Path, timeout: float) -> None:
    """Optional correctness track: how many known seeds this commit still finds."""
    script = repo / "verify_seeds.py"
    if not script.is_file() or c.binary is None:
        return
    r = run([sys.executable, str(script), "-b", str(c.binary),
             "-s", str(seeds_file)], cwd=repo, timeout=timeout)
    m = re.search(r"(\d+)\s+seeds?.*?·\s*(\d+)\s+passed", r.stdout)
    if m:
        c.verify_total = int(m.group(1))
        c.verified = int(m.group(2))


# --------------------------------------------------------------------------
# chart
# --------------------------------------------------------------------------

def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def nice_ticks(lo: float, hi: float, want: int = 6) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / want
    mag = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        step = mag * mult
        if raw <= step:
            break
    first = math.ceil(lo / step) * step
    ticks, v = [], first
    while v <= hi + step * 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


def build_svg(commits: list[Commit], title: str, show_seeds: bool) -> str:
    live = [c for c in commits if c.ok]
    n = len(live)
    col = max(120, min(190, 1400 // max(n, 1)))
    pad_l, pad_r = 96, 40
    width = pad_l + pad_r + col * max(n - 1, 1)
    top, plot_h = 66, 250
    y0, y1 = top, top + plot_h
    lab_y = y1 + 26
    seeds_top = lab_y + 44
    seeds_h = 40
    height = (seeds_top + seeds_h + 34) if show_seeds else (lab_y + 40)

    vals = [c.mean for c in live]
    errs = [c.stderr for c in live]
    vlo = min(v - e for v, e in zip(vals, errs))
    vhi = max(v + e for v, e in zip(vals, errs))
    pad = max((vhi - vlo) * 0.18, vhi * 0.04, 0.5)
    lo, hi = max(0.0, vlo - pad), vhi + pad
    ticks = nice_ticks(lo, hi)
    if ticks:
        lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])

    def X(i: int) -> float:
        return pad_l + col * i

    def Y(v: float) -> float:
        return y1 - (v - lo) * (y1 - y0) / (hi - lo)

    base = vals[0]
    parts: list[str] = []

    # gridlines
    for t in ticks:
        parts.append(f'<line class="gl" x1="{pad_l}" y1="{Y(t):.1f}" '
                     f'x2="{width - pad_r}" y2="{Y(t):.1f}"/>')
        parts.append(f'<text class="tk" x="{pad_l - 12}" y="{Y(t) + 4:.1f}" '
                     f'text-anchor="end">{t:g}</text>')
    parts.append(f'<text class="ax" x="{pad_l}" y="{top - 12}">ms / iteration</text>')

    # error bars, then the line, then the dots on top
    for i, c in enumerate(live):
        if c.stderr > 0:
            a, b = Y(c.mean - c.stderr), Y(c.mean + c.stderr)
            parts.append(f'<line class="eb" x1="{X(i):.1f}" y1="{a:.1f}" '
                         f'x2="{X(i):.1f}" y2="{b:.1f}"/>')
    pts = " ".join(f"{X(i):.1f},{Y(c.mean):.1f}" for i, c in enumerate(live))
    parts.append(f'<polyline class="ln" points="{pts}"/>')
    for i, c in enumerate(live):
        parts.append(f'<circle class="dot" cx="{X(i):.1f}" cy="{Y(c.mean):.1f}" r="5"/>')
        parts.append(f'<text class="vl" x="{X(i):.1f}" y="{Y(c.mean) - 14:.1f}" '
                     f'text-anchor="middle">{c.mean:.2f}</text>')
        if i:
            d = (c.mean / base - 1.0) * 100.0
            parts.append(f'<text class="dl" x="{X(i):.1f}" y="{Y(c.mean) + 21:.1f}" '
                         f'text-anchor="middle">{d:+.1f}%</text>')

    # x labels, staggered when they would collide
    for i, c in enumerate(live):
        dy = 0 if (col >= 150 or i % 2 == 0) else 17
        subj = c.subject if len(c.subject) <= 26 else c.subject[:25] + "…"
        parts.append(f'<text class="xs" x="{X(i):.1f}" y="{lab_y + dy}" '
                     f'text-anchor="middle">{esc(c.short)}</text>')
        parts.append(f'<text class="xl" x="{X(i):.1f}" y="{lab_y + dy + 14}" '
                     f'text-anchor="middle">{esc(subj)}</text>')

    # correctness track
    if show_seeds:
        tot = max((c.verify_total or 0) for c in live) or 1
        sy0, sy1 = seeds_top, seeds_top + seeds_h

        def SY(k: int) -> float:
            return sy1 - (k / tot) * (sy1 - sy0)

        parts.append(f'<text class="ax" x="{pad_l}" y="{seeds_top - 12}">'
                     f'seeds found</text>')
        for t in (0, tot):
            parts.append(f'<line class="gl" x1="{pad_l}" y1="{SY(t):.1f}" '
                         f'x2="{width - pad_r}" y2="{SY(t):.1f}"/>')
            parts.append(f'<text class="tk" x="{pad_l - 12}" y="{SY(t) + 4:.1f}" '
                         f'text-anchor="end">{t}</text>')
        got = [(i, c) for i, c in enumerate(live) if c.verified is not None]
        for (ia, ca), (ib, cb) in zip(got, got[1:]):
            parts.append(f'<line class="ds" x1="{X(ia):.1f}" y1="{SY(ca.verified):.1f}" '
                         f'x2="{X(ib):.1f}" y2="{SY(cb.verified):.1f}"/>')
        for i, c in got:
            cls = "dd bad" if c.verified < (c.verify_total or tot) else "dd"
            parts.append(f'<circle class="{cls}" cx="{X(i):.1f}" '
                         f'cy="{SY(c.verified):.1f}" r="4"/>')
            parts.append(f'<text class="sl" x="{X(i):.1f}" '
                         f'y="{SY(c.verified) + 20:.1f}" text-anchor="middle">'
                         f'{c.verified}/{c.verify_total}</text>')

    css = []
    for sel, d in (("", LIGHT), ("@media (prefers-color-scheme: dark)", DARK)):
        body = (
            f".bgr{{fill:{d['bg']}}}"
            f".gl{{stroke:{d['grid']};stroke-width:1}}"
            f".tk,.xs{{fill:{d['faint']}}}"
            f".ax{{fill:{d['faint']}}}"
            f".ln{{fill:none;stroke:{d['accent']};stroke-width:2;"
            f"stroke-linejoin:round;stroke-linecap:round}}"
            f".eb{{stroke:{d['accent']};stroke-width:1;opacity:.45}}"
            f".dot{{fill:{d['accent']};stroke:{d['card']};stroke-width:2}}"
            f".vl{{fill:{d['ink']}}}"
            f".dl{{fill:{d['accent']}}}"
            f".xl{{fill:{d['ink2']}}}"
            f".ds{{stroke:{d['muted']};stroke-width:1.5}}"
            f".dd{{fill:{d['muted']};stroke:{d['card']};stroke-width:1.5}}"
            f".dd.bad{{fill:{d['warn']}}}"
            f".sl{{fill:{d['muted']}}}"
            f".ttl{{fill:{d['ink']}}}"
        )
        css.append(body if not sel else f"{sel}{{{body}}}")
    base_css = (
        "text{font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace}"
        ".tk,.xs,.sl,.dl{font-size:11px}"
        ".ax{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase}"
        ".vl{font-size:12.5px;font-weight:600}"
        ".xl{font-size:11.5px;font-family:system-ui,-apple-system,sans-serif}"
        ".ttl{font-size:15px;font-weight:600;"
        "font-family:system-ui,-apple-system,sans-serif}"
        "text{font-variant-numeric:tabular-nums}"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{esc(title)}">'
        f"<style>{base_css}{''.join(css)}</style>"
        f'<rect class="bgr" x="0" y="0" width="{width}" height="{height}"/>'
        f'<text class="ttl" x="{pad_l - 12}" y="24">{esc(title)}</text>'
        f"{''.join(parts)}</svg>"
    )


def build_html(commits: list[Commit], svg: str, meta: dict) -> str:
    live = [c for c in commits if c.ok]
    base = live[0].mean if live else math.nan
    rows = []
    for c in commits:
        if not c.ok:
            rows.append(f"<tr><td>{esc(c.short)}</td><td>{esc(c.subject)}</td>"
                        f'<td colspan="4" class="bad">build or run failed</td></tr>')
            continue
        d = "" if c is live[0] else f"{(c.mean / base - 1) * 100:+.2f}%"
        seeds = (f"{c.verified}/{c.verify_total}"
                 if c.verified is not None else "—")
        rows.append(
            f"<tr><td class=m>{esc(c.short)}</td><td>{esc(c.subject)}</td>"
            f"<td class='m n'>{c.mean:.3f}</td><td class='m n'>"
            f"{('&plusmn;%.3f' % c.stderr) if len(c.runs) > 1 else '&mdash;'}</td>"
            f"<td class='m n {'good' if d.startswith('-') else ''}'>{d}</td>"
            f"<td class='m n'>{seeds}</td></tr>")
    info = " &middot; ".join(
        f"{k} <b>{v}</b>" for k, v in meta.items())
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Commit benchmark</title>
<style>
:root{{color-scheme:light dark}}
body{{margin:0;padding:32px 24px 64px;background:{LIGHT['bg']};color:{LIGHT['ink']};
 font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
.wrap{{max-width:1200px;margin:0 auto;display:flex;flex-direction:column;gap:26px}}
h1{{font-size:26px;margin:0;letter-spacing:-.02em}}
p.sub{{margin:0;color:{LIGHT['muted']};max-width:70ch}}
.card{{background:{LIGHT['card']};border:1px solid {LIGHT['line']};border-radius:10px;
 overflow-x:auto;padding:14px}}
table{{width:100%;border-collapse:collapse;font-size:13.5px}}
th{{text-align:left;font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;
 color:{LIGHT['faint']};font-weight:500;padding:10px 12px;
 border-bottom:1px solid {LIGHT['line']};white-space:nowrap}}
td{{padding:9px 12px;border-bottom:1px solid {LIGHT['grid']};vertical-align:top}}
tr:last-child td{{border-bottom:none}}
.m{{font-family:ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}}
.n{{text-align:right;white-space:nowrap}}
.good{{color:{LIGHT['accent']}}} .bad{{color:{LIGHT['warn']}}}
footer{{color:{LIGHT['faint']};font-size:12.5px;border-top:1px solid {LIGHT['line']};
 padding-top:16px}}
@media (prefers-color-scheme:dark){{
 body{{background:{DARK['bg']};color:{DARK['ink']}}}
 p.sub{{color:{DARK['muted']}}}
 .card{{background:{DARK['card']};border-color:{DARK['line']}}}
 th{{color:{DARK['faint']};border-bottom-color:{DARK['line']}}}
 td{{border-bottom-color:{DARK['grid']}}}
 .good{{color:{DARK['accent']}}} .bad{{color:{DARK['warn']}}}
 footer{{color:{DARK['faint']};border-top-color:{DARK['line']}}}}}
</style></head><body><div class="wrap">
<h1>Commit benchmark</h1>
<p class="sub">Each commit built and timed round-robin rather than one after
another, so a warming GPU cannot flatter whichever ran first. Error bars are one
standard error; differences smaller than a couple of those are noise.</p>
<div class="card">{svg}</div>
<div class="card" style="padding:0">
<table><thead><tr><th>Commit</th><th>Subject</th><th class="n">ms/iter</th>
<th class="n">&plusmn;se</th><th class="n">vs first</th><th class="n">Seeds</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<footer>{info}</footer>
</div></body></html>"""


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build and benchmark a range of commits, then chart ms/iteration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Two things")[0].strip())
    where = ap.add_mutually_exclusive_group()
    where.add_argument("-n", "--commits", type=int, default=None, metavar="N",
                       help="chart HEAD~N through HEAD (default 5 if --from is absent)")
    where.add_argument("--from", "--since", dest="from_commit", default=None,
                       metavar="COMMIT",
                       help="chart from this commit through HEAD; takes any revision "
                            "git understands (hash, tag, branch, HEAD~7). If it is "
                            "not an ancestor of HEAD, their merge base is used")
    ap.add_argument("--max-commits", type=int, default=40, metavar="M",
                    help="refuse ranges longer than this (default 40); every commit "
                         "is built and timed, so long ranges take hours")
    ap.add_argument("--no-first-parent", action="store_true",
                    help="follow every parent through merges instead of just the "
                         "mainline")
    ap.add_argument("-r", "--runs", type=int, default=4,
                    help="measured rounds per commit (default 4)")
    ap.add_argument("-w", "--warmup", type=int, default=2,
                    help="minimum leading rounds to discard (default 2)")
    ap.add_argument("--warmup-runs", type=int, default=10, metavar="R",
                    help="discard at least this many runs in total, whatever the "
                         "commit count, so the GPU reaches steady state (default 10)")
    ap.add_argument("-s", "--seed", default=None,
                    help="start seed (default: first entry of seeds.txt)")
    ap.add_argument("-p", "--print-interval", type=int, default=256,
                    help="iterations per --benchmark run (default 256)")
    ap.add_argument("--arch", default=None,
                    help="GPU arch passed to make as ARCH=, e.g. sm_75")
    ap.add_argument("-D", "--make-arg", action="append", default=[],
                    metavar="VAR=VAL", help="extra make argument (repeatable)")
    ap.add_argument("-P", "--build-workers", type=int, default=None, metavar="W",
                    help="build this many commits at once, each in its own git "
                         "worktree (default: cpu count / 4, capped at 4). Every "
                         "worker runs a full nvcc, so raise it with an eye on RAM")
    ap.add_argument("-j", "--jobs", type=int, default=None, metavar="J",
                    help="make -j per worker (default: cpu count / workers)")
    ap.add_argument("-o", "--out", default="bench_out", help="output directory")
    ap.add_argument("--verify", action="store_true",
                    help="also run verify_seeds.py per commit and chart the result")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="per-run timeout in seconds (default 1800)")
    ap.add_argument("--title", default=None, help="chart title")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="arguments after -- are passed to the benchmark binary")
    args = ap.parse_args()
    extra = [a for a in args.rest if a != "--"]

    repo = Path(__file__).resolve().parent
    if not (repo / ".git").exists():
        top = run(["git", "-C", str(repo), "rev-parse", "--show-toplevel"])
        if top.returncode != 0:
            sys.exit("not a git repository")
        repo = Path(top.stdout.strip())

    seed = args.seed
    if seed is None:
        for cand in (repo / "seeds.txt", repo.parent / "seeds.txt"):
            if cand.is_file():
                for raw in cand.read_text().splitlines():
                    line = raw.split("#", 1)[0].strip()
                    if line:
                        seed = line
                        break
            if seed:
                break
    if seed is None:
        seed = "-9149722601043664674"

    make_args = list(args.make_arg)
    if args.arch:
        make_args.append(f"ARCH={args.arch}")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # One benchmark at a time: two of these sharing a GPU makes every number
    # meaningless, and they would fight over the worktree as well.
    lock = out / ".bench.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        sys.exit(f"another run appears to be active ({lock}); remove it if stale")

    if args.commits is None and args.from_commit is None:
        args.commits = 5
    commits = resolve_commits(repo, args.commits, args.from_commit,
                              not args.no_first_parent, args.max_commits)
    stage = out / "bin"
    stage.mkdir(exist_ok=True)
    workers = args.build_workers or max(1, min(4, (os.cpu_count() or 4) // 4))
    workers = max(1, min(workers, len(commits)))
    jobs = args.jobs or max(1, (os.cpu_count() or 4) // workers)
    worktrees: list[Path] = []
    started = time.monotonic()

    try:
        print(f"repo   : {repo}")
        print(f"seed   : {seed}")
        print(f"commits: {len(commits)}  ({commits[0].short} .. {commits[-1].short})")
        print(f"rounds : {args.warmup} warmup + {args.runs} measured, round-robin")
        print(f"build  : {workers} worker(s) x make -j{jobs}")
        print(f"output : {out}\n")

        # One worktree per worker, created up front. git worktree add touches
        # shared metadata, so they are made serially; the checkouts afterwards
        # are per-worktree and safe to run at the same time.
        for i in range(workers):
            wt = Path(tempfile.mkdtemp(prefix=f"bench_wt{i}_"))
            wt.rmdir()   # git wants to create the directory itself
            git(repo, "worktree", "add", "--detach", "--force",
                str(wt), commits[0].sha)
            worktrees.append(wt)

        free: queue.Queue[Path] = queue.Queue()
        for wt in worktrees:
            free.put(wt)
        say = threading.Lock()
        done = [0]

        def build_one(c: Commit) -> None:
            wt = free.get()
            try:
                t0 = time.monotonic()
                ok = build_commit(wt, c, stage, make_args, args.print_interval, jobs)
                dt = time.monotonic() - t0
                with say:
                    done[0] += 1
                    print(f"  [{done[0]:>2}/{len(commits)}] {c.short}  "
                          f"{'built' if ok else 'FAILED':6} {dt:5.0f}s  "
                          f"{c.subject[:52]}", flush=True)
                if not ok:
                    (out / f"build-fail-{c.short}.log").write_text(c.build_log)
            finally:
                free.put(wt)

        print("building")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(build_one, commits))
        print(f"  built in {time.monotonic() - started:.0f}s")

        buildable = [c for c in commits if c.binary]
        if len(buildable) < 2:
            sys.exit("fewer than two commits built; nothing to chart")

        # Warm up by runs, not rounds. A GPU reaching steady state is what makes
        # the measured rounds comparable, and a short range would otherwise be
        # timed cold: on this hardware the same binary reads 15% faster at 40C
        # than at the 79C it settles to.
        warmup_rounds = max(args.warmup,
                            -(-args.warmup_runs // max(len(buildable), 1)))
        if warmup_rounds != args.warmup:
            print(f"  (warmup raised to {warmup_rounds} rounds = "
                  f"{warmup_rounds * len(buildable)} runs, to reach steady state)")
        total_rounds = warmup_rounds + args.runs
        print(f"\ntiming ({total_rounds} rounds x {len(buildable)} commits)")
        for rnd in range(1, total_rounds + 1):
            measured = rnd > warmup_rounds
            for c in buildable:
                sink = Commit(c.sha, c.short, c.subject, c.binary)
                target = c if measured else sink
                good = time_once(target, seed, args.print_interval, out,
                                 args.timeout, extra)
                if not good and measured:
                    print(f"  round {rnd} {c.short}: run failed", flush=True)
            tag = "measured" if measured else "warmup  "
            print(f"  round {rnd}/{total_rounds} {tag} "
                  f"({time.monotonic() - started:.0f}s elapsed)", flush=True)

        if args.verify:
            seeds_file = next((p for p in (repo / "seeds.txt", repo.parent / "seeds.txt")
                               if p.is_file()), None)
            if seeds_file is None:
                print("\n--verify: no seeds.txt found, skipping")
            else:
                print("\nverifying")
                for c in buildable:
                    verify_commit(repo, c, seeds_file, args.timeout * 4)
                    print(f"  {c.short}: {c.verified}/{c.verify_total}", flush=True)

        gpu_note = ""
        probe = run(["nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm",
                     "--format=csv,noheader"])
        if probe.returncode == 0 and probe.stdout.strip():
            gpu_note = probe.stdout.strip().splitlines()[0]
            print(f"\ngpu at end of run: {gpu_note}")

        # ---- report ----
        live = [c for c in commits if c.ok]
        scanned = {c.seeds_scanned for c in live}
        base = live[0].mean
        print(f"\n{'commit':>9}  {'ms/iter':>9} {'±se':>7} {'vs first':>9}  subject")
        for c in live:
            d = "" if c is live[0] else f"{(c.mean / base - 1) * 100:+.2f}%"
            se = f"{c.stderr:7.3f}" if len(c.runs) > 1 else "      -"
            print(f"{c.short:>9}  {c.mean:9.3f} {se} {d:>9}  {c.subject[:48]}")
        if len(scanned) > 1:
            print("\n  warning: commits scanned different seed counts "
                  f"({sorted(scanned)}); ms/iteration is not directly comparable.")

        title = args.title or (f"{live[0].short} → {live[-1].short}"
                               f"  ({len(live)} commits)")
        svg = build_svg(commits, title, args.verify and
                        any(c.verified is not None for c in live))
        meta = {
            "seed": seed,
            "PRINT_INTERVAL": args.print_interval,
            "rounds": f"{args.runs} measured, {warmup_rounds} discarded",
            "protocol": "round-robin",
        }
        if gpu_note:
            meta["gpu at finish"] = gpu_note
        (out / "chart.svg").write_text(svg)
        (out / "chart.html").write_text(build_html(commits, svg, meta))
        with (out / "results.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sha", "short", "subject", "ms_per_iter", "stderr",
                        "runs", "seeds_scanned", "final_outputs", "seeds_found"])
            for c in commits:
                w.writerow([c.sha, c.short, c.subject,
                            f"{c.mean:.4f}" if c.ok else "",
                            f"{c.stderr:.4f}" if c.ok else "",
                            len(c.runs), c.seeds_scanned, c.final_out,
                            "" if c.verified is None else c.verified])
        (out / "results.json").write_text(json.dumps(
            {"seed": seed, "print_interval": args.print_interval,
             "warmup_rounds": warmup_rounds, "runs": args.runs,
             "gpu": gpu_note,
             "commits": [{"sha": c.sha, "short": c.short, "subject": c.subject,
                          "ms_per_iter": None if not c.ok else c.mean,
                          "stderr": c.stderr, "samples": c.runs,
                          "seeds_scanned": c.seeds_scanned,
                          "seeds_found": c.verified} for c in commits]},
            indent=1))

        print(f"\nwrote {out/'chart.svg'}")
        print(f"      {out/'chart.html'}")
        print(f"      {out/'results.csv'}  {out/'results.json'}")
        return 0

    finally:
        for wt in worktrees:
            run(["git", "-C", str(repo), "worktree", "remove", "--force", str(wt)])
            shutil.rmtree(wt, ignore_errors=True)
        run(["git", "-C", str(repo), "worktree", "prune"])
        for f in out.glob("*.out"):
            f.unlink(missing_ok=True)
        try:
            lock.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
