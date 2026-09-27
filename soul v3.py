#!/usr/bin/env python3
"""
soul_v3.py - the Soul, rebuilt: every stock's character and its memory of
similar moments, fresh every day, strictly from finished trades.

    python soul_v3.py build --root %CACHE_DAILY_ROOT%

Writes <panel>/soul/soul_v3.parquet (timestamp, symbol, soul_* columns).
soul_test.py picks it up automatically.

TWO BLOCKS
==========
PERSONALITY - how has THIS stock behaved?
  Recency-weighted record (1-year half-life) of its trades: target-first rate,
  stop-first rate, realised bracket return, MFE, MAE. Each is SHRUNK toward the
  market's record by its effective sample size: 12 overlapping trades say
  little, 400 say a lot.
      shrunk = (n_eff * stock + K * market) / (n_eff + K)       K = 30

MEMORY - what followed moments LIKE today?
  Similarity is measured in the canonical 8-dimension stock state plus
  drawdown and ATR-z (the families that survived every fold), all per-date
  ranks, so "similar" means similar RELATIVE to peers.
    self   the stock's own 20 nearest past moments
    cross  the 50 nearest moments across all stocks
  Closer analogues count more (weight 1 / (distance + 0.1)). The self answer
  is shrunk toward the cross answer by how many self analogues exist.

TIMING - THE ONE RULE
---------------------
A trade entered on day d is finished at the close of d+5. So at day T:
  * personality uses trades entered at least 6 of that stock's sessions
    earlier (liquidity gaps only make this more conservative);
  * the market prior uses entry days at least 6 sessions earlier;
  * memory pools are rebuilt monthly and contain only trades entered at least
    6 sessions before the month's first session.
Tested by poisoning every label from a date onward: every Soul value dated
before that date + 6 sessions must be identical, to the last digit.
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

CODE_VERSION = "soul_v3"
CFG = {"lag": 6, "halflife_days": 365, "prior_k": 30.0, "self_shrink_k": 10.0,
       "k_self": 20, "k_cross": 50, "min_self": 30, "cross_pool": 200_000, "min_pool": 5_000,
       "min_personality": 20, "kernel_eps": 0.1, "horizon": 5, "seed": 7}
OUTCOMES = {"tp": "label_tp_before_sl", "ret": "label_exit_ret", "mfe": "label_mfe_5d",
            "mae": "label_mae_5d"}                         # plus sl = first_touch == -1
EXTRA_DIMS = ("D_drawdown_252", "D_atr_pct_z252")


def _log(msg):
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def _naive(s):
    t = pd.to_datetime(s)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    return t


def load(pp: Path):
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(pp).schema_arrow.names)
    need = ["timestamp", "symbol", "label_first_touch"] + list(OUTCOMES.values())
    st = [c for _, c, _ in RC.STOCK_STATE_SPEC if c in have]
    ex = [c for c in EXTRA_DIMS if c in have]
    p = pd.read_parquet(pp, columns=need + st + ex)
    p["timestamp"] = _naive(p["timestamp"])
    return p.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def similarity_space(p: pd.DataFrame, date_code: np.ndarray):
    X, names, missing = RC.stock_state(
        lambda c: pd.to_numeric(p[c], errors="coerce").to_numpy() if c in p.columns else None, date_code)
    cols = [X]
    for c in EXTRA_DIMS:
        if c in p.columns:
            r = pd.Series(pd.to_numeric(p[c], errors="coerce").to_numpy()).groupby(date_code).rank(pct=True) - 0.5
            cols.append(r.to_numpy("float32")[:, None])
    Z = np.hstack(cols).astype("float32")
    miss = np.isnan(Z).sum(axis=1).astype("int8")
    return np.nan_to_num(Z, nan=0.0), miss


def personality(p, sym, date_code, n_sessions, cfg) -> pd.DataFrame:
    """Recency-weighted, lagged, shrunk per-stock record."""
    lag, hl = cfg["lag"], f"{cfg['halflife_days']}D"
    ft = pd.to_numeric(p["label_first_touch"], errors="coerce")
    outs = {k: pd.to_numeric(p[c], errors="coerce") for k, c in OUTCOMES.items()}
    outs["sl"] = (ft == -1).astype(float).where(ft.notna())
    ts = p["timestamp"]
    out = pd.DataFrame(index=p.index)
    g_known = None
    for k, s in outs.items():
        known = s.groupby(sym).shift(lag)                          # finished trades only
        if g_known is None:
            g_known = known.notna().astype(float).groupby(sym).cumsum()
        ew = pd.Series(np.nan, index=p.index)
        for _, idx in known.groupby(sym).groups.items():
            ks = known.loc[idx]
            if ks.notna().sum() == 0:
                continue
            ew.loc[idx] = ks.ewm(halflife=hl, times=ts.loc[idx], ignore_na=True).mean().to_numpy()
        # market prior: EW over ENTRY days of the day's mean outcome, lagged in sessions
        daily = s.groupby(date_code).mean().reindex(range(n_sessions))
        prior_d = daily.ewm(halflife=max(cfg["halflife_days"] * 250 / 365, 1), ignore_na=True).mean().shift(lag)
        prior = prior_d.reindex(date_code).to_numpy()
        # effective sample: overlapping 5-day trades, recency-capped
        cap = 2.0 * cfg["halflife_days"] * 250 / 365
        n_eff = np.minimum(g_known.to_numpy(), cap) / cfg["horizon"]
        shr = (n_eff * ew.to_numpy() + cfg["prior_k"] * prior) / (n_eff + cfg["prior_k"])
        shr = np.where(g_known.to_numpy() >= cfg["min_personality"], shr, np.nan)
        out[f"soul_p_{k}"] = shr.astype("float32")
        if k in ("tp", "ret"):
            out[f"soul_p_{k}_vs_market"] = (shr - prior).astype("float32")
    n_eff = np.minimum(g_known.to_numpy(), 2.0 * cfg["halflife_days"] * 250 / 365) / cfg["horizon"]
    out["soul_p_neff"] = n_eff.astype("float32")
    return out


def memory(p, sym, date_code, Z, cfg, log=None) -> pd.DataFrame:
    """Monthly-refreshed analogue pools; daily queries; distance-weighted; shrunk."""
    from sklearn.neighbors import NearestNeighbors
    lag = cfg["lag"]
    ft = pd.to_numeric(p["label_first_touch"], errors="coerce").to_numpy()
    Y = np.column_stack([pd.to_numeric(p[c], errors="coerce").to_numpy() for c in OUTCOMES.values()]
                        + [(ft == -1).astype(float)])
    ok = np.isfinite(Y).all(axis=1)
    names = list(OUTCOMES) + ["sl"]
    n = len(p)
    res = {f"soul_m_self_{k}": np.full(n, np.nan, "float32") for k in names}
    res.update({f"soul_m_cross_{k}": np.full(n, np.nan, "float32") for k in names})
    for k in ("self_n", "self_dist", "cross_dist"):
        res[f"soul_m_{k}"] = np.full(n, np.nan, "float32")
    months = p["timestamp"].dt.to_period("M").to_numpy()
    month_ids, m_inv = np.unique(months, return_inverse=True)
    first_code = pd.Series(date_code).groupby(m_inv).min().to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    # per-symbol row lists, in date order (rows are already sorted by date)
    sym_rows = pd.Series(np.arange(n)).groupby(sym).apply(np.asarray).to_dict()
    sym_codes = {s: date_code[r] for s, r in sym_rows.items()}
    eps = cfg["kernel_eps"]
    for mi in range(len(month_ids)):
        cut = first_code[mi] - lag                              # pool: entered <= cut
        pool = np.where(ok & (date_code <= cut))[0]
        q = np.where(m_inv == mi)[0]
        if len(pool) < cfg["min_pool"] or not len(q):
            continue
        # cross analogues
        cp = pool if len(pool) <= cfg["cross_pool"] else np.sort(rng.choice(pool, cfg["cross_pool"], replace=False))
        k = min(cfg["k_cross"], len(cp))
        nn = NearestNeighbors(n_neighbors=k).fit(Z[cp])
        dist, ind = nn.kneighbors(Z[q])
        w = 1.0 / (dist + eps)
        w /= w.sum(axis=1, keepdims=True)
        for j, nm in enumerate(names):
            res[f"soul_m_cross_{nm}"][q] = (w * Y[cp][ind, j]).sum(axis=1)
        res["soul_m_cross_dist"][q] = dist.mean(axis=1)
        # self analogues, per symbol
        qs = pd.Series(q).groupby(sym[q]).apply(np.asarray)
        for s, qi in qs.items():
            rows, codes = sym_rows[s], sym_codes[s]
            upto = np.searchsorted(codes, cut, side="right")
            own = rows[:upto]
            own = own[ok[own]]
            if len(own) < cfg["min_self"]:
                continue
            kk = min(cfg["k_self"], len(own))
            D = np.sqrt(np.maximum(((Z[qi][:, None, :] - Z[own][None, :, :]) ** 2).sum(-1), 0))
            part = np.argpartition(D, kk - 1, axis=1)[:, :kk]
            dd = np.take_along_axis(D, part, axis=1)
            ww = 1.0 / (dd + eps)
            ww /= ww.sum(axis=1, keepdims=True)
            for j, nm in enumerate(names):
                res[f"soul_m_self_{nm}"][qi] = (ww * Y[own][part, j]).sum(axis=1)
            res["soul_m_self_n"][qi] = float(len(own))
            res["soul_m_self_dist"][qi] = dd.mean(axis=1)
        if log and (mi % 12 == 0):
            log(f"    memory: {month_ids[mi]} ({len(pool):,} finished trades in the pool)")
    out = pd.DataFrame(res, index=p.index)
    n_self = out["soul_m_self_n"].fillna(0).to_numpy() / cfg["horizon"]
    for nm in names:
        sv, cv = out[f"soul_m_self_{nm}"].to_numpy(), out[f"soul_m_cross_{nm}"].to_numpy()
        blend = np.where(np.isfinite(sv), (n_self * sv + cfg["self_shrink_k"] * cv) /
                         (n_self + cfg["self_shrink_k"]), cv)
        out[f"soul_m_{nm}"] = blend.astype("float32")               # the shrunk answer
    out["soul_m_gap_tp"] = (out["soul_m_self_tp"] - out["soul_m_cross_tp"]).astype("float32")
    out["soul_m_gap_ret"] = (out["soul_m_self_ret"] - out["soul_m_cross_ret"]).astype("float32")
    tp = out["soul_m_tp"].to_numpy()
    out["soul_m_tp_se"] = np.sqrt(np.clip(tp * (1 - tp), 0, None) /
                                  np.maximum(n_self + cfg["k_cross"] / cfg["horizon"], 1)).astype("float32")
    return out


def build(root: Path, overrides: dict | None = None, verbose: bool = True,
          panel_frame: pd.DataFrame | None = None) -> Path:
    t0 = time.perf_counter()
    cfg = {**CFG, **(overrides or {})}
    pp = root / "panel" / "panel.parquet"
    p = panel_frame.copy() if panel_frame is not None else load(pp)
    sessions = np.sort(p["timestamp"].unique())
    date_code = np.searchsorted(sessions, p["timestamp"].to_numpy())
    sym = pd.factorize(p["symbol"])[0]
    log = _log if verbose else None
    if verbose:
        _log(f"soul v3: {len(p):,} rows, {p['symbol'].nunique():,} symbols, "
             f"{pd.Timestamp(sessions[0]).date()} -> {pd.Timestamp(sessions[-1]).date()}")
    Z, miss = similarity_space(p, date_code)
    if verbose:
        _log("personality block")
    P1 = personality(p, sym, date_code, len(sessions), cfg)
    if verbose:
        _log("memory block (monthly pools, daily queries)")
    P2 = memory(p, sym, date_code, Z, cfg, log)
    out = pd.concat([p[["timestamp", "symbol"]], P1, P2], axis=1)
    out["soul_state_missing"] = miss
    out = out[out.filter(like="soul_").drop(columns="soul_state_missing").notna().any(axis=1)]
    RC.assert_no_label_leak([c for c in out.columns if c.startswith("soul_")], "soul_v3")
    sd = root / "panel" / "soul"
    sd.mkdir(parents=True, exist_ok=True)
    fp = sd / "soul_v3.parquet"
    out.to_parquet(fp, index=False)
    (sd / "soul_v3_meta.json").write_text(json.dumps({
        "code": CODE_VERSION, "built_at": dt.datetime.now().isoformat(), "config": cfg,
        "rows": int(len(out)), "first": str(out["timestamp"].min().date()) if len(out) else None,
        "last": str(out["timestamp"].max().date()) if len(out) else None,
        "features": [c for c in out.columns if c.startswith("soul_")]}, indent=2), encoding="utf-8")
    if verbose:
        _log(f"wrote {fp} - {len(out):,} rows, {sum(c.startswith('soul_') for c in out.columns)} "
             f"features ({(time.perf_counter()-t0)/60:.1f} min)")
    return fp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT") or "")
    if not str(root):
        raise SystemExit("CACHE_DAILY_ROOT not set and --root not given")
    build(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
