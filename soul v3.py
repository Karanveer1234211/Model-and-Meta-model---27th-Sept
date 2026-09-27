#!/usr/bin/env python3
"""
soul_v3.py - the Soul, rebuilt: every stock's character and its memory of
similar moments, fresh every day, strictly from finished trades.

    python soul_v3.py build --root %CACHE_DAILY_ROOT%
    python soul_v3.py cards --root %CACHE_DAILY_ROOT%          (reference cards for the watchlist)

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

CODE_VERSION = "soul_v3.1"
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


# ======================================================================
# SOUL CARDS - reference memory for the stocks on the watchlist
# ======================================================================
CONTEXT_RAW = {"D_rsi14": ("RSI 14", "{:.0f}"), "D_adx14": ("ADX 14", "{:.0f}"),
               "D_ema20_angle_deg": ("EMA20 angle", "{:+.1f} deg"),
               "D_realvol_20": ("realised vol 20d", "{:.2f}"),
               "D_realvol_ratio_20_60": ("vol ratio 20d/60d", "{:.2f}"),
               "D_dvol_z20": ("volume z-score 20d", "{:+.1f}"),
               "D_pos_in_52w_range": ("52-week range position", "{:.2f}"),
               "D_drawdown_252": ("drawdown from 252d high", "{:.3f}"),
               "D_atr_pct": ("ATR %", "{:.2f}"), "D_atr_pct_z252": ("ATR z-score 252d", "{:+.1f}"),
               "D_intraday_ret_pct": ("today's intraday move %", "{:+.2f}"),
               "D_gap_pct": ("today's gap %", "{:+.2f}")}
EXTRA_WORDS = {"D_drawdown_252": "drawdown", "D_atr_pct_z252": "ATR-z"}
CARD_LABELS = ["label_first_touch", "label_tp_before_sl", "label_exit_ret", "label_mfe_5d",
               "label_mae_5d", "label_days_to_tp", "label_days_to_sl"]


def _dim_names(p):
    names = [n[3:] for n in RC.STOCK_STATE_NAMES
             if dict(zip(RC.STOCK_STATE_NAMES, [c for _, c, _ in RC.STOCK_STATE_SPEC]))[n] in p.columns]
    return names + [EXTRA_WORDS[c] for c in EXTRA_DIMS if c in p.columns]


def _word(pct):
    return "very high" if pct >= 90 else "high" if pct >= 75 else "very low" if pct <= 10 else \
        "low" if pct <= 25 else "middle"


def _outcome(ft):
    return {1.0: "TARGET", -1.0: "STOP", 0.0: "TIMEOUT"}.get(float(ft), "-") if np.isfinite(ft) else "-"


def _summ(rows: pd.DataFrame) -> dict:
    if rows is None or not len(rows):
        return {"n": 0}
    ft = rows["label_first_touch"]
    hit = rows[ft == 1]
    return {"n": int(len(rows)), "target_first": float((ft == 1).mean()), "stop_first": float((ft == -1).mean()),
            "timeout": float((ft == 0).mean()), "avg_ret": float(rows["label_exit_ret"].mean()),
            "median_mfe": float(rows["label_mfe_5d"].median()), "median_mae": float(rows["label_mae_5d"].median()),
            "avg_days_to_target": float(hit["label_days_to_tp"].mean()) if len(hit) else float("nan")}


def soul_cards(root: Path, day: str | None = None, symbols: list | None = None, watch: pd.DataFrame | None = None,
               k_self: int = 10, k_cross: int = 10, cross_summary_k: int = 50, verbose: bool = True,
               console_top: int = 3, cfg: dict | None = None) -> dict:
    """Detailed reference card per watchlist stock. Descriptive only - never used for any decision."""
    from sklearn.neighbors import NearestNeighbors
    cfg = {**CFG, **(cfg or {})}
    lag = cfg["lag"]
    pp = root / "panel" / "panel.parquet"
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(pp).schema_arrow.names)
    st = [c for _, c, _ in RC.STOCK_STATE_SPEC if c in have]
    cols = ["timestamp", "symbol", "close"] + [c for c in CARD_LABELS if c in have] + st + \
        [c for c in EXTRA_DIMS if c in have] + [c for c in CONTEXT_RAW if c in have and c not in st]
    p = pd.read_parquet(pp, columns=list(dict.fromkeys(cols)))
    p["timestamp"] = _naive(p["timestamp"])
    p = p.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    sessions = np.sort(p["timestamp"].unique())
    code = np.searchsorted(sessions, p["timestamp"].to_numpy())
    Z, miss = similarity_space(p, code)
    dims = _dim_names(p)
    d0 = pd.Timestamp(day) if day else pd.Timestamp(sessions[-1])
    d0_code = int(np.searchsorted(sessions, np.datetime64(d0)))
    eng = root / "panel" / "engine"
    if watch is None and symbols is None:
        wf = eng / f"watchlist_{d0.date()}.csv"
        if not wf.exists():
            raise SystemExit(f"no watchlist for {d0.date()} - run the watchlist first or pass symbols")
        watch = pd.read_csv(wf)
    if symbols is None:
        symbols = watch["symbol"].tolist()
    cut = d0_code - lag                               # finished trades only
    lab_ok = np.isfinite(p[["label_first_touch", "label_exit_ret"]].to_numpy(float)).all(axis=1)
    pool = np.where(lab_ok & (code <= cut))[0]
    market = p.loc[lab_ok, "label_exit_ret"].groupby(code[lab_ok]).mean()
    rng = np.random.default_rng(cfg["seed"])
    cp = pool if len(pool) <= cfg["cross_pool"] else np.sort(rng.choice(pool, cfg["cross_pool"], replace=False))
    nn = NearestNeighbors(n_neighbors=min(cross_summary_k, len(cp))).fit(Z[cp]) if len(cp) else None
    soul_row = None
    sf = root / "panel" / "soul" / "soul_v3.parquet"
    if sf.exists():
        sv = pd.read_parquet(sf)
        sv["timestamp"] = _naive(sv["timestamp"])
        soul_row = sv[sv["timestamp"] == d0].set_index("symbol")
    vp = eng / "validation_picks.parquet"
    V = pd.read_parquet(vp) if vp.exists() else None
    if V is not None:
        V["timestamp"] = _naive(V["timestamp"])
    led = eng / "paper_ledger.csv"
    Lg = pd.read_csv(led) if led.exists() else None

    cards = []
    for sym in symbols:
        today = np.where((code == d0_code) & (p["symbol"].to_numpy() == sym))[0]
        if not len(today):
            continue
        i = today[0]
        z = Z[i]
        state = [{"dim": dn, "pct": float((z[j] + 0.5) * 100)} for j, dn in enumerate(dims)]
        raw = {lab: fmt.format(float(p.loc[i, c])) for c, (lab, fmt) in CONTEXT_RAW.items()
               if c in p.columns and pd.notna(p.loc[i, c])}
        c = p["close"].to_numpy(float)
        own_all = np.where(p["symbol"].to_numpy() == sym)[0]
        prior = own_all[code[own_all] <= d0_code]
        if len(prior) > 5:
            raw["5-session return"] = f"{c[prior[-1]] / c[prior[-6]] - 1:+.2%}"
        # SELF memory: declustered nearest past moments of this stock
        own = own_all[lab_ok[own_all] & (code[own_all] <= cut)]
        self_rows, d_random = pd.DataFrame(), float("nan")
        if len(own):
            dist = np.sqrt(((Z[own] - z) ** 2).sum(axis=1))
            d_random = float(np.median(dist))
            picked, used = [], []
            for j in np.argsort(dist):
                cj = code[own[j]]
                if all(abs(cj - u) >= 5 for u in used):
                    picked.append(j); used.append(cj)
                if len(picked) >= k_self:
                    break
            idx = own[picked]
            self_rows = p.loc[idx, ["timestamp"] + [c_ for c_ in CARD_LABELS if c_ in p.columns]].copy()
            self_rows["distance"] = dist[picked]
            self_rows["market_ret"] = [market.get(code[j_], np.nan) for j_ in idx]
        own_hist = p.loc[own, [c_ for c_ in CARD_LABELS if c_ in p.columns]] if len(own) else None
        # CROSS memory
        cross_rows, cross_summary = pd.DataFrame(), {"n": 0}
        if nn is not None:
            dd, ii = nn.kneighbors(z[None, :])
            near = cp[ii[0]]
            cross_summary = _summ(p.loc[near, [c_ for c_ in CARD_LABELS if c_ in p.columns]])
            top = near[:k_cross]
            cross_rows = p.loc[top, ["timestamp", "symbol"] + [c_ for c_ in CARD_LABELS if c_ in p.columns]].copy()
            cross_rows["distance"] = dd[0][:k_cross]
        # previous watchlist appearances (engine's unseen-data picks + forward ledger)
        prev = pd.DataFrame()
        if V is not None:
            vv = V[(V["symbol"] == sym) & (V["rk"] <= 10) & (V["timestamp"] < d0)]
            if len(vv):
                prev = vv[["timestamp", "rk", "label_first_touch", "label_exit_ret"]].rename(
                    columns={"rk": "rank"}).assign(source="walk-forward")
        if Lg is not None and "symbol" in Lg:
            lg = Lg[(Lg["symbol"] == sym)].copy()
            if len(lg):
                lg["timestamp"] = pd.to_datetime(lg["date"])
                lg = lg[lg["timestamp"] < d0].merge(
                    p[["timestamp", "symbol", "label_first_touch", "label_exit_ret"]],
                    on=["timestamp", "symbol"], how="left")
                prev = pd.concat([prev, lg[["timestamp", "rank", "label_first_touch", "label_exit_ret"]]
                                  .assign(source="paper ledger")], ignore_index=True)
        wrow = watch[watch["symbol"] == sym].iloc[0].to_dict() if watch is not None and sym in set(watch["symbol"]) else {}
        cards.append({"symbol": sym, "date": str(d0.date()), "watch": wrow, "state": state, "raw": raw,
                      "state_missing": int(miss[i]),
                      "personality": (soul_row.loc[sym].to_dict() if soul_row is not None and sym in soul_row.index else {}),
                      "self_rows": self_rows, "self_summary": _summ(self_rows), "stock_overall": _summ(own_hist),
                      "distance_random_day": d_random, "cross_rows": cross_rows, "cross_summary": cross_summary,
                      "previous": prev.sort_values("timestamp") if len(prev) else prev,
                      "previous_summary": {"n": int(len(prev)),
                                           "resolved": int(prev["label_exit_ret"].notna().sum()) if len(prev) else 0,
                                           "target_first": float((prev["label_first_touch"] == 1).mean()) if len(prev) else float("nan"),
                                           "avg_ret": float(prev["label_exit_ret"].mean()) if len(prev) else float("nan")}})
    eng.mkdir(parents=True, exist_ok=True)
    hp = eng / f"soul_cards_{d0.date()}.html"
    hp.write_text(_cards_html(cards, d0, cfg), encoding="utf-8")
    rows = []
    for cd in cards:
        for kind, df in (("self", cd["self_rows"]), ("cross", cd["cross_rows"])):
            if len(df):
                rows.append(df.assign(card_symbol=cd["symbol"], kind=kind))
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(eng / f"soul_analogs_{d0.date()}.csv", index=False)
    if verbose:
        _print_cards(cards[:console_top], hp)
    return {"cards": cards, "html": hp, "cutoff_session": str(pd.Timestamp(sessions[cut]).date()) if cut >= 0 else None}


def _pc(x, f="{:.0%}"):
    return f.format(x) if x is not None and np.isfinite(x) else "-"


def _print_cards(cards, hp):
    print("\n" + "=" * 96)
    print("  SOUL MEMORY - reference only (descriptive; not used for any decision)")
    print("=" * 96)
    for cd in cards:
        w = cd["watch"]
        head = f"  #{w.get('rank', '?')} {cd['symbol']}"
        if w:
            head += f"  | entry {w.get('entry')} target {w.get('target')} stop {w.get('stop')}"
        print(head)
        st = sorted(cd["state"], key=lambda s: -abs(s["pct"] - 50))[:5]
        print("    state now: " + ", ".join(f"{s['dim']} {s['pct']:.0f}th pct ({_word(s['pct'])})" for s in st))
        rw = cd["raw"]
        keys = [k for k in ("RSI 14", "ADX 14", "52-week range position", "drawdown from 252d high",
                            "ATR %", "volume z-score 20d", "5-session return") if k in rw]
        print("    readings:  " + " | ".join(f"{k} {rw[k]}" for k in keys))
        ss, so, cs = cd["self_summary"], cd["stock_overall"], cd["cross_summary"]
        if ss["n"]:
            print(f"    same situation, this stock ({ss['n']} past moments): target first {_pc(ss['target_first'])}, "
                  f"stop first {_pc(ss['stop_first'])}, avg return {_pc(ss['avg_ret'], '{:+.2%}')}, "
                  f"median best {_pc(ss['median_mfe'], '{:+.1%}')} / worst {_pc(ss['median_mae'], '{:+.1%}')}")
            if so.get("n"):
                print(f"      vs this stock overall ({so['n']} trades): target first {_pc(so['target_first'])}, "
                      f"avg return {_pc(so['avg_ret'], '{:+.2%}')}")
        else:
            print("    same situation, this stock: not enough finished history")
        if cs.get("n"):
            print(f"    same situation, all stocks (nearest {cs['n']}): target first {_pc(cs['target_first'])}, "
                  f"avg return {_pc(cs['avg_ret'], '{:+.2%}')}")
        ps = cd["previous_summary"]
        if ps["n"]:
            print(f"    previous watchlist appearances: {ps['n']} ({ps['resolved']} resolved), target first "
                  f"{_pc(ps['target_first'])}, avg return {_pc(ps['avg_ret'], '{:+.2%}')}")
    print(f"\n  full cards (all watchlist stocks, every past moment listed): {hp}")


def _cards_html(cards, d0, cfg):
    import html as H
    css = ("body{font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1040px;margin:0 auto;"
           "padding:20px 14px 60px;color:#1d1d1f;background:#fff}@media(prefers-color-scheme:dark){body{"
           "color:#f2f2f7;background:#111113}.card{border-color:#333!important}}h1{font-size:22px}"
           "h2{font-size:18px;margin:0 0 6px}h3{font-size:14px;margin:16px 0 4px}.mut{opacity:.7}"
           ".card{border:1px solid #ddd;border-radius:12px;padding:14px 16px;margin:18px 0}"
           ".wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}"
           "td,th{padding:4px 8px;border-bottom:1px solid rgba(128,128,128,.25);text-align:right;white-space:nowrap}"
           "td:first-child,th:first-child{text-align:left}.bar{height:8px;border-radius:4px;background:#2563eb}"
           ".T{color:#16a34a}.S{color:#dc2626}.box{border-left:3px solid #f59e0b;padding:8px 12px;"
           "background:rgba(245,158,11,.08)}")
    def pc(x, f="{:.0%}"):
        return _pc(x, f)
    def tbl(df, with_symbol=False):
        if df is None or not len(df):
            return "<p class='mut'>none</p>"
        h = ("<tr><th>date</th>" + ("<th>stock</th>" if with_symbol else "") +
             "<th>distance</th><th>outcome</th><th>days</th><th>return</th><th>best</th><th>worst</th>"
             + ("<th>market that week</th>" if "market_ret" in df else "") + "</tr>")
        body = []
        for _, r in df.iterrows():
            oc = _outcome(r.get("label_first_touch", np.nan))
            days = r.get("label_days_to_tp") if oc == "TARGET" else r.get("label_days_to_sl") if oc == "STOP" else 5
            cls = "T" if oc == "TARGET" else "S" if oc == "STOP" else ""
            body.append(f"<tr><td>{pd.Timestamp(r['timestamp']).date()}</td>"
                        + (f"<td>{H.escape(str(r['symbol']))}</td>" if with_symbol else "")
                        + f"<td>{r['distance']:.2f}</td><td class='{cls}'>{oc}</td>"
                        f"<td>{'' if days is None or not np.isfinite(days) else int(days)}</td>"
                        f"<td>{pc(r.get('label_exit_ret'), '{:+.2%}')}</td><td>{pc(r.get('label_mfe_5d'), '{:+.1%}')}</td>"
                        f"<td>{pc(r.get('label_mae_5d'), '{:+.1%}')}</td>"
                        + (f"<td>{pc(r.get('market_ret'), '{:+.2%}')}</td>" if "market_ret" in df else "") + "</tr>")
        return f"<div class='wrap'><table>{h}{''.join(body)}</table></div>"
    def summ_row(lab, s):
        if not s or not s.get("n"):
            return f"<tr><td>{lab}</td><td colspan='7' class='mut'>not enough history</td></tr>"
        return (f"<tr><td>{lab}</td><td>{s['n']}</td><td>{pc(s['target_first'])}</td><td>{pc(s['stop_first'])}</td>"
                f"<td>{pc(s['timeout'])}</td><td>{pc(s['avg_ret'], '{:+.2%}')}</td>"
                f"<td>{pc(s['median_mfe'], '{:+.1%}')} / {pc(s['median_mae'], '{:+.1%}')}</td>"
                f"<td>{pc(s['avg_days_to_target'], '{:.1f}')}</td></tr>")
    L = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
         f"initial-scale=1'><title>Soul cards {d0.date()}</title><style>{css}</style></head><body>",
         f"<h1>Soul memory - watchlist {d0.date()}</h1>",
         "<div class='box'>Reference only. Nothing on this page is used by the engine or the meta-model. "
         f"Every past moment shown is a trade that had finished before this date (entered at least "
         f"{cfg['lag']} sessions earlier). Small samples: read patterns, not single rows.</div>"]
    for cd in cards:
        w = cd["watch"]
        L.append("<div class='card'>")
        L.append(f"<h2>#{w.get('rank', '?')} {H.escape(cd['symbol'])}</h2>")
        if w:
            L.append(f"<p class='mut'>entry {w.get('entry')} | target {w.get('target')} | stop {w.get('stop')} | "
                     f"stretch {w.get('stretch')} | confidence {w.get('confidence')}% | "
                     f"rank history {w.get('hist_net_bp')} bp</p>")
        L.append("<h3>Current state (percentile versus all stocks today)</h3><div class='wrap'><table>"
                 "<tr><th>dimension</th><th>percentile</th><th></th><th>reading</th></tr>" + "".join(
                     f"<tr><td>{s['dim']}</td><td>{s['pct']:.0f}</td><td style='width:40%'><div class='bar' "
                     f"style='width:{max(2, s['pct']):.0f}%'></div></td><td>{_word(s['pct'])}</td></tr>"
                     for s in cd["state"]) + "</table></div>")
        if cd["state_missing"]:
            L.append(f"<p class='mut'>{cd['state_missing']} state input(s) missing today (treated as middle).</p>")
        L.append("<div class='wrap'><table>" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                                                       for k, v in cd["raw"].items()) + "</table></div>")
        pers = cd["personality"]
        if pers:
            L.append("<h3>Personality (Soul v3: recency-weighted, shrunk toward the market)</h3><p>"
                     f"target first {pc(pers.get('soul_p_tp'))} ({pc(pers.get('soul_p_tp_vs_market'), '{:+.1%}')} vs market) | "
                     f"stop first {pc(pers.get('soul_p_sl'))} | avg return {pc(pers.get('soul_p_ret'), '{:+.2%}')} "
                     f"({pc(pers.get('soul_p_ret_vs_market'), '{:+.2%}')} vs market) | evidence n_eff "
                     f"{pc(pers.get('soul_p_neff'), '{:.0f}')}</p>")
        L.append("<h3>Same situation - summary</h3><div class='wrap'><table><tr><th></th><th>n</th>"
                 "<th>target first</th><th>stop first</th><th>timeout</th><th>avg return</th>"
                 "<th>median best / worst</th><th>avg days to target</th></tr>"
                 + summ_row("this stock, similar moments", cd["self_summary"])
                 + summ_row("this stock, all its trades", cd["stock_overall"])
                 + summ_row("all stocks, similar moments", cd["cross_summary"]) + "</table></div>")
        L.append(f"<h3>This stock's most similar past moments</h3><p class='mut'>distance 0 = identical state; "
                 f"a random past day of this stock is typically {pc(cd['distance_random_day'], '{:.2f}')} away. "
                 f"Spaced at least 5 sessions apart.</p>" + tbl(cd["self_rows"]))
        L.append("<h3>Most similar moments across all stocks</h3>" + tbl(cd["cross_rows"], with_symbol=True))
        prev, ps = cd["previous"], cd["previous_summary"]
        L.append(f"<h3>Previous watchlist appearances</h3>")
        if ps["n"]:
            L.append(f"<p>{ps['n']} appearances ({ps['resolved']} resolved): target first {pc(ps['target_first'])}, "
                     f"avg return {pc(ps['avg_ret'], '{:+.2%}')}.</p><div class='wrap'><table><tr><th>date</th>"
                     "<th>rank</th><th>source</th><th>outcome</th><th>return</th></tr>" + "".join(
                         f"<tr><td>{pd.Timestamp(r['timestamp']).date()}</td><td>{int(r['rank'])}</td>"
                         f"<td>{r['source']}</td><td>{_outcome(r['label_first_touch'])}</td>"
                         f"<td>{pc(r['label_exit_ret'], '{:+.2%}')}</td></tr>" for _, r in prev.iterrows())
                     + "</table></div>")
        else:
            L.append("<p class='mut'>none on record</p>")
        L.append("</div>")
    L.append("</body></html>")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "cards"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--date", default=None, help="cards: watchlist date (default: latest)")
    ap.add_argument("--symbols", default=None, help="cards: comma-separated symbols (default: that day's watchlist)")
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT") or "")
    if not str(root):
        raise SystemExit("CACHE_DAILY_ROOT not set and --root not given")
    if a.cmd == "build":
        build(root)
    else:
        soul_cards(root, day=a.date, symbols=a.symbols.split(",") if a.symbols else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
