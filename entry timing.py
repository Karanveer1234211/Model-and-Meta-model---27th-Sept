#!/usr/bin/env python3
"""
entry_timing.py - does the edge survive the entry you can actually get?

    python entry_timing.py --root %CACHE_DAILY_ROOT%              (build labels + test)
    python entry_timing.py --root %CACHE_DAILY_ROOT% --rebuild    (recompute labels)

THE PROBLEM
===========
Every backtest so far assumed entry at the CLOSE of the signal day. The
watchlist is produced at 23:45 - after that close - so the real entry is the
next session. If part of the edge happens overnight, it cannot be captured.

THREE VERSIONS OF EVERY PICK (same days, same bracket sizes)
    A  reference  entry at the signal day's close; exits fill exactly at the
                  target / stop  (what every backtest assumed; reproduces the
                  panel's label_exit_ret EXACTLY - a built-in check)
    B  next open  entry at the next session's open; barrier fills
    C  realistic  entry at the next session's open; GAP-AWARE fills: if a later
                  session OPENS beyond the stop you are filled at that open
                  (worse than -1 ATR), beyond the target at that open (better)
  Bracket sizes are fixed at the signal: TP = +1.5 x ATR14/close, SL = -1.0 x
  ATR14/close (the label's simple 14-day ATR), applied to the entry price.
  B and C hold through the same five sessions as A (T+1 .. T+5), exiting at
  the close of T+5 if neither barrier is hit; same-session ties count as stop.

THE RULE, FIXED BEFORE IT RUNS
------------------------------
The edge SURVIVES realistic entry only if version C's daily top-3 net return
(35 bp cost) has a 95% block-bootstrap interval above zero.

Picks are the signal engine's own WALK-FORWARD picks (validation_picks.parquet,
every prediction made before its day). Each version's "buy everything"
baseline uses the same entry rule.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import research_common as RC  # noqa: E402

CODE_VERSION = "entry_timing v1"
CFG = {"tp_mult": 1.5, "sl_mult": 1.0, "horizon": 5, "atr_n": 14, "cost_bps": 35.0,
       "top_n": (1, 3, 5, 10), "seed": 7}
VERSIONS = {"A": "signal-day close (reference)", "B": "next open, barrier fills",
            "C": "next open, gap-aware fills (REALISTIC)"}


def _log(msg):
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def _naive(s):
    t = pd.to_datetime(s)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return t.dt.normalize()


def symbol_labels(o, h, l, c, cfg=CFG) -> dict:
    """Outcomes of all three entry versions for every bar of one stock."""
    o, h, l, c = (np.asarray(x, dtype="float64") for x in (o, h, l, c))
    n, H = len(c), cfg["horizon"]
    out = {k: np.full(n, np.nan) for k in ("xr_A", "ft_A", "xr_B", "ft_B", "xr_C", "ft_C", "gap1")}
    if n <= H + cfg["atr_n"]:
        return out
    tr = np.full(n, np.nan)
    tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    atr = pd.Series(tr).rolling(cfg["atr_n"], min_periods=cfg["atr_n"]).mean().to_numpy()
    idx = np.arange(0, n - H)
    a = atr[idx] / c[idx]
    tp, sl = cfg["tp_mult"] * a, cfg["sl_mult"] * a
    K = idx[:, None] + np.arange(1, H + 1)[None, :]
    Hh, Ll, Oo = h[K], l[K], o[K]
    ok = np.isfinite(tp) & np.isfinite(sl) & (c[idx] > 0)

    def barrier(entry):
        with np.errstate(invalid="ignore", divide="ignore"):
            up = Hh / entry[:, None] - 1 >= tp[:, None]
            dn = Ll / entry[:, None] - 1 <= -sl[:, None]
        k_up = np.where(up.any(1), up.argmax(1), H)
        k_dn = np.where(dn.any(1), dn.argmax(1), H)
        res = np.where((k_up < H) & (k_up < k_dn), 1.0, np.where(k_dn < H, -1.0, 0.0))   # tie -> stop
        with np.errstate(invalid="ignore", divide="ignore"):
            xr = np.where(res == 1, tp, np.where(res == -1, -sl, c[idx + H] / entry - 1))
        return res, xr

    eA = c[idx]
    rA, xA = barrier(eA)
    eB = o[idx + 1]
    okB = ok & np.isfinite(eB) & (eB > 0)
    rB, xB = barrier(eB)
    # C: gap-aware. Session 1 is the entry session (entered AT its open): intraday
    # only. From session 2 on, an open beyond a barrier fills at that open.
    with np.errstate(invalid="ignore", divide="ignore"):
        g = Oo / eB[:, None] - 1
        up = Hh / eB[:, None] - 1 >= tp[:, None]
        dn = Ll / eB[:, None] - 1 <= -sl[:, None]
    ev = np.zeros((len(idx), H))            # 1 target, -1 stop, 0 nothing
    rv = np.full((len(idx), H), np.nan)
    for k in range(H):
        if k == 0:
            e_k = np.where(dn[:, 0], -1.0, np.where(up[:, 0], 1.0, 0.0))
            r_k = np.where(dn[:, 0], -sl, np.where(up[:, 0], tp, np.nan))
        else:
            gd, gu = g[:, k] <= -sl, g[:, k] >= tp
            e_k = np.where(gd, -1.0, np.where(gu, 1.0, np.where(dn[:, k], -1.0, np.where(up[:, k], 1.0, 0.0))))
            r_k = np.where(gd, g[:, k], np.where(gu, g[:, k],
                           np.where(dn[:, k], -sl, np.where(up[:, k], tp, np.nan))))
        ev[:, k], rv[:, k] = e_k, r_k
    hit = ev != 0
    kstar = np.where(hit.any(1), hit.argmax(1), H)
    rC = np.where(kstar < H, ev[np.arange(len(idx)), np.minimum(kstar, H - 1)], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xC = np.where(kstar < H, rv[np.arange(len(idx)), np.minimum(kstar, H - 1)], c[idx + H] / eB - 1)
    out["xr_A"][idx], out["ft_A"][idx] = np.where(ok, xA, np.nan), np.where(ok, rA, np.nan)
    out["xr_B"][idx], out["ft_B"][idx] = np.where(okB, xB, np.nan), np.where(okB, rB, np.nan)
    out["xr_C"][idx], out["ft_C"][idx] = np.where(okB, xC, np.nan), np.where(okB, rC, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["gap1"][idx] = np.where(okB, eB / c[idx] - 1, np.nan)
    return out


def labels_for(root: Path, keys: pd.DataFrame, cfg=CFG, log=None) -> pd.DataFrame:
    """Entry-timing outcomes for the given (timestamp, symbol) rows, from the raw cache."""
    from data_quality import _paths
    keys = keys[["timestamp", "symbol"]].drop_duplicates()
    parts = []
    syms = sorted(keys["symbol"].unique())
    for j, s in enumerate(syms):
        fp, _ = _paths(root, s)
        if not Path(fp).exists():
            continue
        d = pd.read_parquet(fp, columns=["timestamp", "open", "high", "low", "close"])
        d["timestamp"] = _naive(d["timestamp"])
        d = d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        lab = symbol_labels(d["open"], d["high"], d["low"], d["close"], cfg)
        f = pd.DataFrame({"timestamp": d["timestamp"], "symbol": s, **lab})
        want = set(keys.loc[keys["symbol"] == s, "timestamp"])
        parts.append(f[f["timestamp"].isin(want)])
        if log and (j + 1) % 250 == 0:
            log(f"    labels: {j+1}/{len(syms)} symbols")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build(root: Path, rebuild: bool = False, cfg=CFG, verbose=True) -> pd.DataFrame:
    out = root / "panel" / "entry_timing"
    out.mkdir(parents=True, exist_ok=True)
    fp = out / "entry_labels.parquet"
    if fp.exists() and not rebuild:
        return pd.read_parquet(fp)
    P = pd.read_parquet(root / "panel" / "panel.parquet", columns=["timestamp", "symbol"])
    P["timestamp"] = _naive(P["timestamp"])
    if verbose:
        _log(f"computing entry-timing outcomes for {len(P):,} panel rows from the raw cache")
    L = labels_for(root, P, cfg, _log if verbose else None)
    L.to_parquet(fp, index=False)
    return L


def run(root: Path, rebuild: bool = False, cfg=None, verbose=True) -> dict:
    t0 = time.perf_counter()
    cfg = {**CFG, **(cfg or {})}
    L = build(root, rebuild, cfg, verbose)
    cost = cfg["cost_bps"] / 1e4
    # built-in check: version A must reproduce the panel's own label
    P = pd.read_parquet(root / "panel" / "panel.parquet", columns=["timestamp", "symbol", "label_exit_ret"])
    P["timestamp"] = _naive(P["timestamp"])
    M = P.merge(L[["timestamp", "symbol", "xr_A"]], on=["timestamp", "symbol"], how="inner")
    both = M["label_exit_ret"].notna() & M["xr_A"].notna()
    ref_diff = float(np.abs(M.loc[both, "label_exit_ret"] - M.loc[both, "xr_A"]).max()) if both.any() else float("nan")
    vp = root / "panel" / "engine" / "validation_picks.parquet"
    if not vp.exists():
        raise SystemExit("no engine walk-forward picks - run: python signal_engine.py train")
    V = pd.read_parquet(vp, columns=["timestamp", "symbol", "rk"])
    V["timestamp"] = _naive(V["timestamp"])
    V = V[V["rk"] <= max(cfg["top_n"])].merge(L, on=["timestamp", "symbol"], how="left")
    days = set(V["timestamp"])
    U = L[L["timestamp"].isin(days)]
    res = {"code": CODE_VERSION, "built_at": dt.datetime.now().isoformat(), "cost_bps": cfg["cost_bps"],
           "reference_check_max_diff": ref_diff, "reference_rows_checked": int(both.sum()),
           "days": int(V["timestamp"].nunique()), "versions": {}}
    day_series = {}
    for v in VERSIONS:
        uni = (U[f"xr_{v}"] - cost).groupby(U["timestamp"]).mean()
        res["versions"][v] = {"universe": RC.block_bootstrap_mean(uni.to_numpy(), seed=cfg["seed"]), "topn": {}}
        for n in cfg["top_n"]:
            t = (V.loc[V["rk"] <= n, f"xr_{v}"] - cost).groupby(V.loc[V["rk"] <= n, "timestamp"]).mean()
            ex = (t - uni.reindex(t.index)).dropna()
            res["versions"][v]["topn"][int(n)] = {
                "net": RC.block_bootstrap_mean(t.to_numpy(), seed=cfg["seed"]),
                "excess": RC.block_bootstrap_mean(ex.to_numpy(), seed=cfg["seed"])}
            if n == 3:
                day_series[v] = t
    for v in ("B", "C"):
        d = (day_series[v] - day_series["A"].reindex(day_series[v].index)).dropna()
        res[f"cost_of_waiting_{v}"] = RC.block_bootstrap_mean(d.to_numpy(), seed=cfg["seed"])
    # how much of the picks' edge happens overnight (before you can trade)?
    t3 = V[V["rk"] <= 3]
    g_pick = t3.groupby("timestamp")["gap1"].mean()
    g_uni = U.groupby("timestamp")["gap1"].mean()
    res["overnight_gap_bp"] = {"picks": float(g_pick.mean() * 1e4), "universe": float(g_uni.mean() * 1e4),
                               "picks_minus_universe": RC.block_bootstrap_mean(
                                   (g_pick - g_uni.reindex(g_pick.index)).dropna().to_numpy(), seed=cfg["seed"])}
    yr = t3["timestamp"].dt.year
    res["by_year"] = {int(y): {v: float((g[f"xr_{v}"] - cost).groupby(g["timestamp"]).mean().mean() * 1e4)
                               for v in VERSIONS} for y, g in t3.groupby(yr)}
    stop_gaps = t3[(t3["ft_C"] == -1) & (t3["xr_C"] < t3["xr_B"] - 1e-12)]
    res["stop_gap_share"] = float(len(stop_gaps) / max((t3["ft_C"] == -1).sum(), 1))
    c3 = res["versions"]["C"]["topn"][3]["net"]
    res["survives"] = bool(np.isfinite(c3["lo"]) and c3["lo"] > 0)
    res["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    out = root / "panel" / "entry_timing"
    (out / "entry_timing.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    RC.ledger_append(root / "panel" / "panel.parquet", {"kind": "exploration", "tool": CODE_VERSION,
                                                        "survives": res["survives"]})
    if verbose:
        _print(res)
    return res


def _f(e):
    return f"{e['mean']*1e4:+5.0f} [{e['lo']*1e4:+4.0f},{e['hi']*1e4:+4.0f}]"


def _print(res):
    print("\n" + "=" * 84)
    print(f"  ENTRY TIMING - the engine's walk-forward picks, {res['days']} days, net of {res['cost_bps']:.0f} bp")
    print("=" * 84)
    print(f"  check: version A reproduces the panel's labels to {res['reference_check_max_diff']:.1e} "
          f"on {res['reference_rows_checked']:,} rows")
    print(f"\n  {'version':<42}{'top-1 net':>18}{'top-3 net':>18}{'top-3 excess':>18}")
    for v, lab in VERSIONS.items():
        r = res["versions"][v]["topn"]
        print(f"  {v}  {lab:<39}{_f(r[1]['net']):>18}{_f(r[3]['net']):>18}{_f(r[3]['excess']):>18}")
    print(f"  buy everything (same entry rule): " + " | ".join(
        f"{v} {_f(res['versions'][v]['universe'])}" for v in VERSIONS))
    print(f"\n  cost of waiting for the next open (top-3, per day): B minus A {_f(res['cost_of_waiting_B'])} | "
          f"C minus A {_f(res['cost_of_waiting_C'])}")
    og = res["overnight_gap_bp"]
    print(f"  overnight gap, signal close -> next open: picks {og['picks']:+.0f} bp vs market "
          f"{og['universe']:+.0f} bp -> picks minus market {_f(og['picks_minus_universe'])}")
    print(f"  stops filled worse than -1 ATR because the stock opened beyond it: "
          f"{res['stop_gap_share']:.0%} of realistic stop-outs")
    print("\n  by year, top-3 net bp (A / B / C): " + " | ".join(
        f"{y} {d['A']:+.0f}/{d['B']:+.0f}/{d['C']:+.0f}" for y, d in res["by_year"].items()))
    print("\n" + "-" * 84)
    print("  RULE: realistic (C) top-3 net, 95% interval above zero")
    print(f"  RESULT: {'the edge SURVIVES realistic entry' if res['survives'] else 'the edge does NOT survive realistic entry - do not trade it as is'}")
    print("-" * 84)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT") or "")
    if not str(root):
        raise SystemExit("CACHE_DAILY_ROOT not set and --root not given")
    run(root, rebuild=a.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
