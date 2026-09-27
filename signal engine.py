#!/usr/bin/env python3
"""
signal_engine.py - the production model, the daily watchlist, and the forward
paper test, in one place.

    python signal_engine.py train     --root %CACHE_DAILY_ROOT%
    python signal_engine.py watchlist --root %CACHE_DAILY_ROOT%
    python signal_engine.py resolve   --root %CACHE_DAILY_ROOT%
    python signal_engine.py history   --root %CACHE_DAILY_ROOT%

WHAT IT IS
==========
One model on all clean features (M0 - the structure every test favoured),
plus four helper models that describe each pick:

    TP    P(target before stop)   -> RANKS the watchlist (raw score)
    SL    P(stop before target)
    MFE   best price reached within 5 sessions  (%)
    MAE   worst price reached within 5 sessions (%)
    DAYS  sessions to the target, given it is hit first

RULES CARRIED OVER FROM THE RESEARCH
------------------------------------
  * Ranking uses the RAW score. Calibrated probabilities tie large groups of
    stocks (77% of days were once decided alphabetically); calibration is
    used only for the confidence figure.
  * Clean features only: labels and forward columns are firewalled; raw
    price LEVELS of outside series (GOLD_close, CRUDEOIL_close, ...) are
    removed - identical for every stock on a day, they can only act as a
    calendar and drift outside the range the model has seen; any feature that
    is really a stock's price level (per-date rank correlation with close
    >= 0.97) is removed too. Daily returns of outside series stay.
  * Every number on the watchlist is first measured on UNSEEN data
    (walk-forward). A helper model no better than a simple guess is labelled
    "weak" on the watchlist instead of being shown as if it were precise.
  * Target and stop use the LABEL'S OWN ATR (simple 14-day mean true range,
    from the raw cache), exactly as the backtests did.

THE FORWARD TEST
----------------
`watchlist` appends each day's picks to paper_ledger.csv. `resolve` scores
them once their 5 sessions have passed: net return, excess over buying
everything, a NIFTY-hedged version, and the ATR +2.0 bracket - the arms
declared before the first pick. This ledger is the only untouched evidence
left; nothing in it can be revised after the fact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import research_common as RC  # noqa: E402

CODE_VERSION = "signal_engine v4.1"
CFG = {"cost_bps": 35.0, "fit_cap": 400_000, "val_splits": 5, "val_min_train_frac": 0.35,
       "embargo": 5, "seed": 7, "top_n": 10, "ledger_top": 3, "tp_mult": 1.5, "sl_mult": 1.0,
       "stretch_mult": 2.0, "hedge_cost_bps": 3.0, "min_days_for_ci": 40,
       "hgb": {"max_iter": 300, "max_depth": 5, "learning_rate": 0.06,
               "min_samples_leaf": 200, "l2_regularization": 1.0}}
LABELS = ["label_tp_before_sl", "label_first_touch", "label_exit_ret", "label_mfe_5d",
          "label_mae_5d", "label_days_to_tp", "label_days_to_sl"]
STRETCH_LABEL = "label_exit_ret_atr2p0_1p0"
CONTEXT = {"D_pos_in_52w_range": "52w pos", "D_drawdown_252": "drawdown",
           "D_atr_pct_z252": "ATR z"}


def _log(msg):
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def _dir(root: Path) -> Path:
    d = root / "panel" / "engine"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _naive(s):
    t = pd.to_datetime(s)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    return t


# ----------------------------------------------------------------------
# features
# ----------------------------------------------------------------------
def clean_features(panel_path: Path, sample_dates: int = 60, seed: int = 0):
    """All model-safe features minus raw levels. Returns (features, excluded dict)."""
    import pyarrow.parquet as pq
    import panel_build as PB
    schema = pq.ParquetFile(panel_path).schema_arrow.names
    base = [c for c in PB.panel_feature_columns(pd.DataFrame(columns=schema))
            if c not in ("open", "high", "low", "close", "volume")]
    RC.assert_no_label_leak(base, "signal_engine")
    levels = [c for c in base if c.endswith(("_close", "_stale"))
              and not c.startswith(("D_", "X_", "R_", "MKT_"))]
    keep = [c for c in base if c not in levels]
    # price-level proxies: per-date rank correlation with the stock's close
    ts = _naive(pd.read_parquet(panel_path, columns=["timestamp"])["timestamp"])
    dates = np.sort(ts.unique())
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(dates, min(sample_dates, len(dates)), replace=False))
    m = ts.isin(pick).to_numpy()
    s = pd.read_parquet(panel_path, columns=["timestamp", "close"] + keep)[m]
    s["timestamp"] = _naive(s["timestamp"])
    rc = s.groupby("timestamp")["close"].rank()
    proxies = []
    for c in keep:
        v = pd.to_numeric(s[c], errors="coerce")
        if v.notna().mean() < 0.5:
            continue
        rv = v.groupby(s["timestamp"]).rank()
        tmp = pd.DataFrame({"a": rv.to_numpy(), "b": rc.to_numpy(), "t": s["timestamp"].to_numpy()})
        rho = tmp.groupby("t")[["a", "b"]].corr().xs("a", level=1)["b"]
        rv_ = np.abs(rho.to_numpy(dtype=float))
        rv_ = rv_[np.isfinite(rv_)]
        if len(rv_) and float(np.median(rv_)) >= 0.97:
            proxies.append(c)
    keep = [c for c in keep if c not in proxies]
    return keep, {"raw_levels_of_outside_series": levels, "price_level_proxies": proxies}


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------
def _clf(cfg):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(early_stopping=False, random_state=cfg["seed"], **cfg["hgb"])


def _reg(cfg):
    """
    Helpers predict the TYPICAL (median) outcome. v3 predicted the average;
    MFE and MAE are lopsided, so the average sits above the typical move and
    lost to a median guess on absolute error despite ranking well (MFE rank
    correlation +0.30 was labelled 'weak'). Median loss makes the comparison fair.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="absolute_error", early_stopping=False,
                                         random_state=cfg["seed"], **cfg["hgb"])


def fit_models(X: pd.DataFrame, L: pd.DataFrame, idx: np.ndarray, cfg: dict, rng) -> dict:
    """Fit the five models on rows idx (capped). L holds the label columns."""
    def cap(ix):
        return ix if len(ix) <= cfg["fit_cap"] else np.sort(rng.choice(ix, cfg["fit_cap"], replace=False))
    ft = L["label_first_touch"].to_numpy()
    tp_i = cap(idx)
    Xtp = X.iloc[tp_i].to_numpy("float32")
    models = {
        "TP": _clf(cfg).fit(Xtp, L["label_tp_before_sl"].to_numpy()[tp_i].astype(int)),
        "SL": _clf(cfg).fit(Xtp, (ft[tp_i] == -1).astype(int)),
        "MFE": _reg(cfg).fit(Xtp, L["label_mfe_5d"].to_numpy()[tp_i]),
        "MAE": _reg(cfg).fit(Xtp, L["label_mae_5d"].to_numpy()[tp_i]),
    }
    del Xtp
    d_i = idx[(ft[idx] == 1) & np.isfinite(L["label_days_to_tp"].to_numpy()[idx])]
    d_i = cap(d_i)
    models["DAYS"] = _reg(cfg).fit(X.iloc[d_i].to_numpy("float32"),
                                   L["label_days_to_tp"].to_numpy()[d_i])
    return models


def predict(models: dict, Xa: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"raw": models["TP"].predict_proba(Xa)[:, 1],
                         "p_sl": models["SL"].predict_proba(Xa)[:, 1],
                         "mfe_hat": models["MFE"].predict(Xa),
                         "mae_hat": models["MAE"].predict(Xa),
                         "days_hat": np.clip(models["DAYS"].predict(Xa), 1, 5)})


# ----------------------------------------------------------------------
# TRAIN = walk-forward validation, then the final fit
# ----------------------------------------------------------------------
def _spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 50:
        return float("nan")
    return float(pd.Series(a[ok]).rank().corr(pd.Series(b[ok]).rank()))


def train(root: Path, overrides: dict | None = None, verbose: bool = True) -> dict:
    import feasibility_test as FT
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score
    t0 = time.perf_counter()
    cfg = {**CFG, **(overrides or {})}
    pp = root / "panel" / "panel.parquet"
    feats, excluded = clean_features(pp)
    _log(f"{len(feats)} clean features | removed {len(excluded['raw_levels_of_outside_series'])} "
         f"raw levels of outside series, {len(excluded['price_level_proxies'])} price-level proxies")
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(pp).schema_arrow.names)
    lab = [c for c in LABELS + [STRETCH_LABEL] if c in have]
    p = pd.read_parquet(pp, columns=["timestamp", "symbol"] + lab + feats)
    p["timestamp"] = _naive(p["timestamp"])
    p = p.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    L = p[lab].apply(pd.to_numeric, errors="coerce")
    ok = L["label_tp_before_sl"].notna() & L["label_first_touch"].notna() & \
        L["label_exit_ret"].notna() & L["label_mfe_5d"].notna() & L["label_mae_5d"].notna()
    ok = ok.to_numpy()
    ts = p["timestamp"].to_numpy()
    sessions = np.sort(np.unique(ts[ok]))
    X = p[feats]
    rng = np.random.default_rng(cfg["seed"])
    _log(f"{int(ok.sum()):,} labelled rows | {pd.Timestamp(sessions[0]).date()} -> "
         f"{pd.Timestamp(sessions[-1]).date()}")

    # ---- walk-forward validation: every watchlist number measured on unseen data
    sp = FT.splits(sessions, cfg["val_splits"], cfg["embargo"], cfg["val_min_train_frac"])
    oos = []
    for s in sp:
        tf = time.perf_counter()
        tr = np.where(ok & (ts <= s["train_end"]))[0]
        te = np.where(ok & (ts >= s["test_start"]) & (ts <= s["test_end"]))[0]
        mdl = fit_models(X, L, tr, cfg, rng)
        pr = predict(mdl, X.iloc[te].to_numpy("float32"))
        pr["timestamp"], pr["fold"], pr["symbol"] = ts[te], s["fold"], p["symbol"].to_numpy()[te]
        for c in lab:
            pr[c] = L[c].to_numpy()[te]
        oos.append(pr)
        _log(f"validation fold {s['fold']} {pd.Timestamp(s['test_start']).date()}.."
             f"{pd.Timestamp(s['test_end']).date()} ({time.perf_counter()-tf:.0f}s)")
    O = pd.concat(oos, ignore_index=True)
    cost = cfg["cost_bps"] / 1e4
    O["net"] = O["label_exit_ret"] - cost
    if STRETCH_LABEL in O:
        O["net_stretch"] = O[STRETCH_LABEL] - cost
    O["rk"] = O.groupby("timestamp")["raw"].rank(ascending=False, method="first")
    uni = O.groupby("timestamp")["net"].mean()

    # rank-position track record
    rank_stats = {}
    for r in range(1, cfg["top_n"] + 1):
        g = O[O["rk"] == r]
        ex = (g.set_index("timestamp")["net"] - uni).dropna()
        rank_stats[r] = {"days": int(len(g)), "net_bp": float(g["net"].mean() * 1e4),
                         "excess_bp": float(ex.mean() * 1e4),
                         "hit": float(g["label_tp_before_sl"].mean()),
                         "stretch_net_bp": float(g["net_stretch"].mean() * 1e4)
                         if "net_stretch" in g else None}
    top3 = O[O["rk"] <= 3]
    t3d = top3.groupby("timestamp")["net"].mean()
    evidence = {"top3_net": RC.block_bootstrap_mean(t3d.to_numpy(), seed=cfg["seed"]),
                "top3_excess": RC.block_bootstrap_mean((t3d - uni.reindex(t3d.index)).dropna().to_numpy(),
                                                       seed=cfg["seed"]),
                "universe": RC.block_bootstrap_mean(uni.to_numpy(), seed=cfg["seed"])}

    # calibration, measured honestly: fit on the earlier folds, test on the last
    last = O["fold"] == O["fold"].max()
    iso_chk = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
        O.loc[~last, "raw"], O.loc[~last, "label_tp_before_sl"])
    chk = pd.DataFrame({"p": iso_chk.predict(O.loc[last, "raw"]),
                        "y": O.loc[last, "label_tp_before_sl"].to_numpy()})
    chk["b"] = pd.qcut(chk["p"].rank(method="first"), 10, labels=False)
    calib = chk.groupby("b").agg(predicted=("p", "mean"), realised=("y", "mean"), n=("y", "size"))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
        O["raw"], O["label_tp_before_sl"])
    # HISTORY: keep the walk-forward predictions. Each fold's confidence comes
    # from a calibration fitted on EARLIER folds only (fold 1 has none -> NaN).
    O["conf_cf"] = np.nan
    for f in sorted(O["fold"].unique())[1:]:
        e_ = O["fold"] < f
        ic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
            O.loc[e_, "raw"], O.loc[e_, "label_tp_before_sl"])
        O.loc[O["fold"] == f, "conf_cf"] = ic.predict(O.loc[O["fold"] == f, "raw"])
    hist_dir = _dir(root)
    O[O["rk"] <= 20].to_parquet(hist_dir / "validation_picks.parquet", index=False)
    uni.rename("universe_net").to_frame().to_parquet(hist_dir / "validation_universe.parquet")

    # helper-model reliability versus a naive guess
    def rel(name, pred, real, naive):
        ok_ = np.isfinite(pred) & np.isfinite(real)
        rho = _spearman(pred[ok_], real[ok_])
        err_m = float(np.mean(np.abs(pred[ok_] - real[ok_])))
        err_n = float(np.mean(np.abs(naive - real[ok_])))
        return {"spearman": rho, "error_model": err_m, "error_naive": err_n,
                "verdict": "useful" if (rho >= 0.10 and err_m < err_n) else "weak"}
    tr_all = np.where(ok)[0]
    reliability = {
        "MFE": rel("MFE", O["mfe_hat"].to_numpy(), O["label_mfe_5d"].to_numpy(),
                   float(np.nanmedian(L["label_mfe_5d"].to_numpy()[tr_all]))),
        "MAE": rel("MAE", O["mae_hat"].to_numpy(), O["label_mae_5d"].to_numpy(),
                   float(np.nanmedian(L["label_mae_5d"].to_numpy()[tr_all]))),
    }
    hitm = O["label_first_touch"] == 1
    reliability["DAYS"] = rel("DAYS", O.loc[hitm, "days_hat"].to_numpy(),
                              O.loc[hitm, "label_days_to_tp"].to_numpy(),
                              float(np.nanmedian(L.loc[L["label_first_touch"] == 1,
                                                       "label_days_to_tp"])))
    sl_y = (O["label_first_touch"] == -1).astype(int)
    reliability["SL"] = {"auc": float(roc_auc_score(sl_y, O["p_sl"])) if sl_y.nunique() > 1 else float("nan")}
    reliability["SL"]["verdict"] = "useful" if reliability["SL"]["auc"] >= 0.53 else "weak"
    reliability["TP"] = {"auc": float(roc_auc_score(O["label_tp_before_sl"], O["raw"]))}
    to = L["label_first_touch"].to_numpy() == 0
    timeout_mean = float(np.nanmean(L["label_exit_ret"].to_numpy()[to & ok]))

    # ---- META-MODEL on the engine's OWN unseen predictions
    import meta_label_test as ML
    mcfg = {**ML.CFG, "cost_bps": cfg["cost_bps"], **cfg.get("meta", {})}
    mpr = O[["timestamp", "symbol", "raw", "net"] + list(ML.HELPER_COLS)].rename(columns={"raw": "p_raw"})
    _log("meta-model: nested walk-forward on the validation predictions")
    extras = ML.panel_extras(root)
    MD, mfeats, form_hist = ML.meta_features(mpr, extras, mcfg)
    meta_eval, meta_hist = ML.nested_evaluation(MD, mfeats, mcfg, return_frame=True)
    meta_hist.to_parquet(_dir(root) / "meta_history.parquet", index=False)
    meta_model = ML.MetaModel(mcfg).fit(MD, mfeats)
    meta_eval["leans_on"] = ML.leans_on(meta_model)

    # ---- final fit on everything labelled
    _log("final fit on all labelled rows")
    final = fit_models(X, L, np.where(ok)[0], cfg, rng)
    meta = {"code": CODE_VERSION, "built_at": dt.datetime.now().isoformat(), "config": cfg,
            "features": feats, "excluded": excluded,
            "trained_through": str(pd.Timestamp(sessions[-1]).date()),
            "validation": {"folds": [{k: str(v) for k, v in s.items()} for s in sp],
                           "evidence": evidence, "rank_stats": rank_stats,
                           "calibration_last_fold": calib.reset_index().to_dict(orient="records"),
                           "reliability": reliability},
            "timeout_mean_ret": timeout_mean}
    mid = hashlib.sha1(json.dumps({k: meta[k] for k in ("code", "features", "trained_through")},
                                  sort_keys=True).encode()).hexdigest()[:10]
    meta["model_id"] = mid
    out = _dir(root)
    meta["meta_model"] = {k: v for k, v in meta_eval.items() if k not in ("blocks",)}
    meta["meta_model"]["features"] = mfeats
    meta["meta_model"]["cutoff"] = meta_model.tau
    with open(out / "models.pkl", "wb") as fh:
        pickle.dump({"models": final, "calibrator": iso, "meta": meta, "meta_model": meta_model,
                     "meta_cfg": mcfg, "form_hist": form_hist}, fh)
    (out / "engine_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    RC.ledger_append(pp, {"kind": "production_model", "tool": CODE_VERSION, "model_id": mid,
                          "trained_through": meta["trained_through"]})
    meta["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    if verbose:
        _print_train(meta)
    return meta


def _f(e):
    return f"{e['mean']*1e4:+.0f} [{e['lo']*1e4:+.0f},{e['hi']*1e4:+.0f}]"


def _print_train(meta):
    v = meta["validation"]
    print("\n" + "=" * 76)
    print(f"  SIGNAL ENGINE {meta['model_id']} - trained through {meta['trained_through']}, "
          f"{len(meta['features'])} clean features")
    print("=" * 76)
    ex = meta["excluded"]
    if ex["raw_levels_of_outside_series"] or ex["price_level_proxies"]:
        print("  removed as raw levels: " + ", ".join(ex["raw_levels_of_outside_series"][:8])
              + (" ..." if len(ex["raw_levels_of_outside_series"]) > 8 else ""))
        if ex["price_level_proxies"]:
            print("  removed as price-level proxies: " + ", ".join(ex["price_level_proxies"]))
    e = v["evidence"]
    print(f"\n  UNSEEN-DATA CHECK (walk-forward, net of {meta['config']['cost_bps']:.0f} bp)")
    print(f"    top-3 net {_f(e['top3_net'])} bp | excess over buy-everything "
          f"{_f(e['top3_excess'])} bp | buy-everything {_f(e['universe'])} bp")
    print("\n  TRACK RECORD BY RANK (what the watchlist quotes)")
    print(f"    {'rank':>4}{'days':>7}{'net bp':>9}{'excess':>9}{'TP-first':>10}{'ATR+2 net':>11}")
    for r, s in v["rank_stats"].items():
        st = f"{s['stretch_net_bp']:+.0f}" if s["stretch_net_bp"] is not None else "-"
        print(f"    {r:>4}{s['days']:>7}{s['net_bp']:>+9.0f}{s['excess_bp']:>+9.0f}"
              f"{s['hit']:>10.0%}{st:>11}")
    print("\n  CONFIDENCE CALIBRATION (fitted on earlier folds, tested on the last)")
    print("    " + ", ".join(f"{c['predicted']:.2f}->{c['realised']:.2f}"
                             for c in v["calibration_last_fold"]))
    print("\n  HELPER MODELS vs A SIMPLE GUESS (unseen data)")
    rl = v["reliability"]
    print(f"    TP  ranking AUC {rl['TP']['auc']:.3f}")
    print(f"    SL  AUC {rl['SL']['auc']:.3f} -> {rl['SL']['verdict']}")
    for k in ("MFE", "MAE", "DAYS"):
        r = rl[k]
        print(f"    {k:<4} rank corr {r['spearman']:+.3f} | error {r['error_model']:.4f} vs guess "
              f"{r['error_naive']:.4f} -> {r['verdict']}")
    import meta_label_test as ML
    mm = meta.get("meta_model")
    if mm:
        ML.print_meta(mm, title="META-MODEL on this engine's unseen predictions")
    print(f"\n  saved: engine/models.pkl ({meta.get('minutes', '?')} min)")


# ----------------------------------------------------------------------
# WATCHLIST
# ----------------------------------------------------------------------
def _atr_simple(root: Path, sym: str, as_of: pd.Timestamp):
    """The label's ATR: simple 14-day mean true range, from the raw cache."""
    from data_quality import _paths
    fp, _ = _paths(root, sym)
    if not Path(fp).exists():
        return None, None
    d = pd.read_parquet(fp, columns=["timestamp", "high", "low", "close"])
    d["timestamp"] = _naive(d["timestamp"]).dt.normalize()
    d = d[d["timestamp"] <= as_of.normalize()].sort_values("timestamp").tail(40)
    if len(d) < 15:
        return None, None
    h, l, c = (d[k].to_numpy(float) for k in ("high", "low", "close"))
    tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    return float(tr[-14:].mean()), float(c[-1])


def load_engine(root: Path):
    fp = _dir(root) / "models.pkl"
    if not fp.exists():
        raise SystemExit("no trained engine - run: python signal_engine.py train")
    with open(fp, "rb") as fh:
        return pickle.load(fh)


def watchlist(root: Path, as_of: str | None = None, verbose: bool = True,
              write_ledger: bool = True) -> pd.DataFrame:
    E = load_engine(root)
    meta, models, iso = E["meta"], E["models"], E["calibrator"]
    cfg = meta["config"]
    pp = root / "panel" / "panel.parquet"
    feats = meta["features"]
    ts = _naive(pd.read_parquet(pp, columns=["timestamp"])["timestamp"])
    day = pd.Timestamp(as_of) if as_of else ts.max()
    m = (ts == day).to_numpy()
    if not m.any():
        raise SystemExit(f"no panel rows on {day.date()}")
    extra = [c for c in list(CONTEXT) + ["X_turnover_med"] if c not in feats]
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(pp).schema_arrow.names)
    extra = [c for c in extra if c in have]
    d = pd.read_parquet(pp, columns=["timestamp", "symbol", "close"] + feats + extra)[m].reset_index(drop=True)
    pr = predict(models, d[feats].to_numpy("float32"))
    d = pd.concat([d[["symbol", "close"] + [c for c in list(CONTEXT) + ["X_turnover_med"] if c in d]],
                   pr], axis=1)
    d["confidence"] = iso.predict(d["raw"])
    d["rank"] = d["raw"].rank(ascending=False, method="first").astype(int)
    d["pct"] = 1 - (d["rank"] - 1) / len(d)
    d = d.sort_values("rank")
    meta_info = _meta_today(root, E, d, day, ts, feats, cfg)
    top = d.head(cfg["top_n"]).copy()
    rows = []
    rs = meta["validation"]["rank_stats"]
    rel = meta["validation"]["reliability"]
    cost = cfg["cost_bps"] / 1e4
    for _, r in top.iterrows():
        atr, entry = _atr_simple(root, r["symbol"], day)
        entry = entry if entry else float(r["close"])
        a = (atr / entry) if atr else np.nan
        tp_pct, sl_pct = cfg["tp_mult"] * a, cfg["sl_mult"] * a
        p_tp, p_sl = float(r["confidence"]), float(np.clip(r["p_sl"], 0, 1))
        p_none = max(0.0, 1 - p_tp - p_sl)
        ev = p_tp * tp_pct - p_sl * sl_pct + p_none * meta["timeout_mean_ret"] - cost
        hist = rs.get(int(r["rank"])) or rs.get(str(int(r["rank"])))
        rows.append({"rank": int(r["rank"]), "symbol": r["symbol"], "entry": round(entry, 2),
                     "target": round(entry * (1 + tp_pct), 2) if atr else None,
                     "stop": round(entry * (1 - sl_pct), 2) if atr else None,
                     "stretch": round(entry * (1 + cfg["stretch_mult"] * a), 2) if atr else None,
                     "confidence": round(100 * p_tp, 1), "p_stop_first": round(100 * p_sl, 1),
                     "exp_mfe_pct": round(100 * r["mfe_hat"], 2),
                     "exp_mae_pct": round(100 * r["mae_hat"], 2),
                     "exp_days_to_tp": round(float(r["days_hat"]), 1),
                     "model_ev_bp": round(ev * 1e4, 0) if atr else None,
                     "hist_net_bp": round(hist["net_bp"], 0) if hist else None,
                     "hist_excess_bp": round(hist["excess_bp"], 0) if hist else None,
                     "rank_pool": int(r["rank"]),
                     "turnover_cr": round(float(r["X_turnover_med"]) / 1e7, 1)
                     if "X_turnover_med" in r and pd.notna(r["X_turnover_med"]) else None,
                     **{v: (round(float(r[k]), 2) if k in r and pd.notna(r[k]) else None)
                        for k, v in CONTEXT.items()},
                     "score_raw": round(float(r["raw"]), 5)})
    W = pd.DataFrame(rows).drop(columns="rank_pool")
    if meta_info is not None:
        W = W.merge(meta_info["picks"], on="symbol", how="left")
    avoid = d.tail(10)["symbol"].tolist()
    out = _dir(root)
    W.to_csv(out / f"watchlist_{day.date()}.csv", index=False)
    if write_ledger:
        led = out / "paper_ledger.csv"
        pool = int(E.get("meta_cfg", {}).get("rerank_pool", cfg["ledger_top"])) if meta_info is not None else cfg["ledger_top"]
        new = W.head(max(pool, cfg["ledger_top"])).assign(date=str(day.date()), model_id=meta["model_id"],
                                                          logged_at=dt.datetime.now().isoformat())
        if led.exists():
            old = pd.read_csv(led)
            old = old[old["date"] != str(day.date())]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(led, index=False)
    if verbose:
        _print_watchlist(W, day, meta, avoid, rel, len(d), meta_info)
    # SOUL MEMORY: reference cards for every watchlist stock (descriptive only)
    try:
        import soul_v3 as SV
        SV.soul_cards(root, day=str(day.date()), watch=W, verbose=verbose)
    except SystemExit as e:
        if verbose:
            print(f"  (soul cards skipped: {e})")
    except Exception as e:                                   # never break the watchlist
        if verbose:
            print(f"  (soul cards skipped: {type(e).__name__}: {e})")
    return W


def _meta_today(root, E, d, day, ts, feats, cfg):
    """The meta-model's full view of today's top picks, plus today's context."""
    if "meta_model" not in E:
        return None
    import meta_label_test as ML
    mm, mcfg, mmeta = E["meta_model"], E["meta_cfg"], E["meta"]["meta_model"]
    cost = cfg["cost_bps"] / 1e4
    pp = root / "panel" / "panel.parquet"
    tp_model = E["models"]["TP"]
    sessions = np.sort(ts.unique())
    # score recent sessions: form needs FINISHED trades since training; persistence needs 5 prior days
    tt = pd.Timestamp(E["meta"]["trained_through"])
    recent = sessions[max(0, np.searchsorted(sessions, day) - 6):np.searchsorted(sessions, day)]
    need = np.union1d(sessions[(sessions > tt) & (sessions < day)], recent)
    hist = pd.DataFrame(columns=["timestamp", "symbol", "p_raw"])
    form = E["form_hist"].copy()
    if len(need):
        m = ts.isin(set(need)).to_numpy()
        Ld = pd.read_parquet(pp, columns=["timestamp", "symbol", "label_exit_ret"] + feats)[m]
        Ld["timestamp"] = _naive(Ld["timestamp"])
        Ld["p_raw"] = tp_model.predict_proba(Ld[feats].to_numpy("float32"))[:, 1]
        hist = Ld[["timestamp", "symbol", "p_raw"]]
        Ld["rk"] = Ld.groupby("timestamp")["p_raw"].rank(ascending=False, method="first")
        lt = Ld[(Ld["timestamp"] > tt) & (Ld["rk"] <= mcfg["act_top"]) & Ld["label_exit_ret"].notna()]
        if len(lt):
            live = (lt["label_exit_ret"] - cost).groupby(lt["timestamp"]).mean()
            form = pd.concat([form, live]).sort_index()
            form = form[~form.index.duplicated(keep="last")]
    form = form.reindex(sessions[sessions <= day])       # positions = sessions; NaN = unresolved
    today = d[["symbol", "raw"] + list(ML.HELPER_COLS)].rename(columns={"raw": "p_raw"}).assign(timestamp=day)
    pr = pd.concat([hist, today], ignore_index=True)
    ex = ML.panel_extras(root)
    MD, _, _ = ML.meta_features(pr, ex[ex["timestamp"].isin(set(pr["timestamp"]))], mcfg, form_series=form)
    MD = MD[MD["timestamp"] == day].reset_index(drop=True)
    for f in mm.feats:
        if f not in MD:
            MD[f] = np.nan
    p, lo, hi = mm.predict(MD)
    MD["meta_p"] = p
    k, pool = mcfg["act_top"], mcfg.get("rerank_pool", 5)
    skill = mmeta.get("brier_skill", 0) > 0
    out = pd.DataFrame({"symbol": MD["symbol"], "rank_": MD["rank"]})
    out["meta_prob"] = np.where(skill, np.round(100 * p, 1), np.nan)
    out["meta_band"] = [f"{100*a:.0f}-{100*b:.0f}" if skill else "n/a" for a, b in zip(lo, hi)]
    # how picks at a similar meta-score did on unseen days
    rel = mmeta.get("reliability", [])
    if rel:
        pr_ = np.array([r["predicted"] for r in rel])
        out["meta_decile"] = [int(rel[int(np.argmin(np.abs(pr_ - x)))]["decile"]) for x in p]
        out["meta_hist_bp"] = [round(rel[int(np.argmin(np.abs(pr_ - x)))]["net_bp"]) for x in p]
        worst = min(rel, key=lambda r: r["net_bp"])
        out["caution"] = [(d == worst["decile"] and worst["net_bp"] < 0) for d in out["meta_decile"]]
    inpool = MD["rank"] <= pool
    mr = pd.Series(np.nan, index=MD.index)
    mr[inpool] = MD.loc[inpool, "meta_p"].rank(ascending=False, method="first")
    out["meta_rank"] = mr.to_numpy()
    top = (MD["rank"] <= k).to_numpy()
    if mmeta.get("filter_validated"):
        out["action"] = np.where(~top, "-", np.where(p >= mm.tau, "TRADE", "SKIP"))
    else:
        out["action"] = np.where(~top, "-", "TRADE*")
    w = np.clip(p / np.mean(p[top]), 0.5, 1.5) if top.any() else np.ones(len(p))
    out["size_x"] = np.where(top, np.round(w * k / w[top].sum(), 2), np.nan) if top.any() else np.nan
    # context the meta-model leans on
    fr = form.dropna()
    lag = mcfg["form_lag"]
    lagged = form.shift(lag)
    ctx = {f"engine_form_{w_}d_bp": float(lagged.rolling(w_, min_periods=max(3, w_ // 2)).mean().iloc[-1] * 1e4)
           for w_ in mcfg["form_windows"]} if len(form) > lag else {}
    mkd = ex.drop_duplicates("timestamp").set_index("timestamp")
    if "mk_vol20" in mkd and day in mkd.index:
        v = mkd["mk_vol20"]
        ctx["market_vol_percentile"] = float((v <= v.loc[day]).mean())
    if "INDIAVIX_close" in mkd and day in mkd.index:
        ctx["india_vix"] = float(mkd.loc[day, "INDIAVIX_close"])
    return {"picks": out.drop(columns="rank_"), "context": ctx,
            "policies": {"filter": bool(mmeta.get("filter_validated")),
                         "sizing": bool(mmeta.get("sizing_validated")),
                         "rerank": bool(mmeta.get("rerank_validated"))},
            "meta_top3": MD.loc[inpool].sort_values("meta_p", ascending=False).head(k)["symbol"].tolist(),
            "skill": skill}


def _print_watchlist(W, day, meta, avoid, rel, n_universe, meta_info=None):
    stale = (pd.Timestamp.now().normalize() - day.normalize()).days
    print("\n" + "=" * 96)
    print(f"  WATCHLIST {day.date()} | model {meta['model_id']} (trained through "
          f"{meta['trained_through']}) | {n_universe} stocks ranked")
    if stale > 4:
        print(f"  WARNING: latest panel date is {stale} days old - update the panel first")
    print("=" * 96)
    tag = {k: ("" if rel[k]["verdict"] == "useful" else " (weak)") for k in ("MFE", "MAE", "DAYS", "SL")}
    print(f"  {'#':>2} {'symbol':<12}{'entry':>9}{'target':>9}{'stop':>9}{'stretch':>9}"
          f"{'conf':>6}{'P(stop)':>8}{'MFE%':>7}{'MAE%':>7}{'days':>6}{'EV bp':>7}{'hist':>6}{'liq cr':>7}")
    for _, r in W.iterrows():
        print(f"  {r['rank']:>2} {r['symbol']:<12}{r['entry']:>9}{str(r['target']):>9}{str(r['stop']):>9}"
              f"{str(r['stretch']):>9}{r['confidence']:>5.0f}%{r['p_stop_first']:>7.0f}%"
              f"{r['exp_mfe_pct']:>+7.1f}{r['exp_mae_pct']:>+7.1f}{r['exp_days_to_tp']:>6.1f}"
              f"{str(r['model_ev_bp']):>7}{str(r['hist_net_bp']):>6}{str(r['turnover_cr']):>7}")
    if meta_info is not None:
        mv, pol, ctx = meta["meta_model"], meta_info["policies"], meta_info["context"]
        tick = lambda b: "validated" if b else "not validated"
        print("\n  " + "-" * 92)
        print(f"  META VIEW  (policies on unseen days: filter {tick(pol['filter'])} | sizing "
              f"{tick(pol['sizing'])} | re-rank {tick(pol['rerank'])}; AUC {mv['auc']:.3f})")
        c = []
        if ctx.get("engine_form_20d_bp") is not None:
            c.append("engine form (finished trades, bp/trade): " + ", ".join(
                f"{w}d {ctx[f'engine_form_{w}d_bp']:+.0f}" for w in (5, 20, 60)
                if np.isfinite(ctx.get(f"engine_form_{w}d_bp", np.nan))))
        if "market_vol_percentile" in ctx:
            c.append(f"market volatility {ctx['market_vol_percentile']:.0%} percentile")
        if "india_vix" in ctx:
            c.append(f"VIX {ctx['india_vix']:.1f}")
        if c:
            print("  context: " + " | ".join(c))
        print(f"  {'#':>2} {'symbol':<12}{'meta':>7}{'band':>9}{'decile':>8}{'hist bp':>9}"
              f"{'size':>7}{'meta #':>8}  action")
        pool = W[W["meta_rank"].notna()] if "meta_rank" in W else W.head(5)
        for _, r in pool.iterrows():
            mp = f"{r['meta_prob']:.0f}%" if pd.notna(r.get("meta_prob")) else "n/a"
            sz = f"x{r['size_x']:.2f}" if pd.notna(r.get("size_x")) else "-"
            print(f"  {r['rank']:>2} {r['symbol']:<12}{mp:>7}{str(r.get('meta_band', '')):>9}"
                  f"{str(r.get('meta_decile', '-')):>8}{str(r.get('meta_hist_bp', '-')):>9}{sz:>7}"
                  f"{int(r['meta_rank']) if pd.notna(r.get('meta_rank')) else '-':>8}  {r.get('action', '')}"
                  + ("  ! worst meta decile historically" if bool(r.get("caution", False)) else ""))
        print(f"  re-ranked top-3 (best 3 of the engine's top-5 by meta): {', '.join(meta_info['meta_top3'])}"
              + ("  [validated]" if pol["rerank"] else "  [forward-test arm only]"))
        notes = []
        if not meta_info["skill"]:
            notes.append("meta % hidden: its probabilities were not better than the base rate on unseen days")
        notes.append("hist bp = what picks at a similar meta score earned on unseen days")
        wl = mv.get("within_rank_lift_mean_bp")
        if wl is not None and np.isfinite(wl):
            notes.append(f"meta's own edge beyond rank: {wl:+.0f} bp/pick (same rank, higher vs lower meta)")
        if pol["sizing"]:
            notes.append("size = validated capital tilt within the top-3 (sums to 3)")
        else:
            notes.append("size = forward-test arm only (not validated); trade equal size")
        if not pol["filter"]:
            notes.append("TRADE* = trade the plain top-3 (the skip filter is not validated)")
        for n_ in notes:
            print(f"  - {n_}")
        print("  " + "-" * 92)
    print(f"\n  conf = calibrated P(target before stop). hist = what this RANK earned per trade on unseen")
    print(f"  data (net of {meta['config']['cost_bps']:.0f} bp). EV = model estimate from conf, stop odds and "
          f"the bracket - a guide, not a promise.")
    print(f"  helper reliability on unseen data: MFE{tag['MFE']}, MAE{tag['MAE']}, days{tag['DAYS']}, "
          f"P(stop){tag['SL']} - '(weak)' means no better than a simple guess; ignore it.")
    print(f"  target = +{meta['config']['tp_mult']} x ATR, stop = -{meta['config']['sl_mult']} x ATR, "
          f"stretch = +{meta['config']['stretch_mult']} x ATR (the research-favoured wider target); "
          f"exit after 5 sessions if neither hits.")
    print(f"  bottom of today's ranking (historically the worst decile): {', '.join(avoid)}")
    print(f"  top {meta['config']['ledger_top']} logged to engine/paper_ledger.csv for the forward test")


# ----------------------------------------------------------------------
# RESOLVE the forward paper test
# ----------------------------------------------------------------------
def resolve(root: Path, verbose: bool = True) -> dict:
    from data_quality import _paths
    out = _dir(root)
    led = out / "paper_ledger.csv"
    if not led.exists():
        raise SystemExit("no paper ledger yet - run the watchlist first")
    E = load_engine(root)
    cfg = E["meta"]["config"]
    cost = cfg["cost_bps"] / 1e4
    T = pd.read_csv(led)
    T["date"] = pd.to_datetime(T["date"])
    pp = root / "panel" / "panel.parquet"
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(pp).schema_arrow.names)
    lab = [c for c in ["label_exit_ret", STRETCH_LABEL, "label_first_touch", "label_days_to_tp",
                       "label_days_to_sl"] if c in have]
    P = pd.read_parquet(pp, columns=["timestamp", "symbol"] + lab)
    P["timestamp"] = _naive(P["timestamp"]).dt.normalize()
    P = P[P["timestamp"].isin(set(T["date"]))]
    uni = (P.assign(n=P["label_exit_ret"] - cost).groupby("timestamp")["n"].mean())
    J = T.merge(P, left_on=["date", "symbol"], right_on=["timestamp", "symbol"], how="left")
    J = J[J["label_exit_ret"].notna()].copy()
    Jall = J.copy()                                   # top-5 incl. meta columns
    J = J[J["rank"] <= cfg["ledger_top"]].copy()      # PRIMARY = the engine's own top-3
    res = {"logged_days": int(T["date"].nunique()), "resolved_days": int(J["date"].nunique())}
    if J.empty:
        if verbose:
            print(f"\n  {res['logged_days']} days logged, none resolved yet (each needs 5 sessions)")
        return res
    J["net"] = J["label_exit_ret"] - cost
    J["excess"] = J["net"] - J["date"].map(uni)
    if STRETCH_LABEL in J:
        J["net_stretch"] = J[STRETCH_LABEL] - cost
    # NIFTY hedge over each trade's actual holding period
    fp, _ = _paths(root, "NIFTY50")
    if Path(fp).exists():
        nf = pd.read_parquet(fp, columns=["timestamp", "close"])
        nf["timestamp"] = _naive(nf["timestamp"]).dt.normalize()
        nf = nf.drop_duplicates("timestamp").set_index("timestamp")["close"].sort_index()
        pos = {t: i for i, t in enumerate(nf.index)}
        def hold(r):
            k = r["label_days_to_tp"] if r["label_first_touch"] == 1 else (
                r["label_days_to_sl"] if r["label_first_touch"] == -1 else 5)
            i = pos.get(r["date"])
            if i is None or not np.isfinite(k) or i + int(k) >= len(nf):
                return np.nan
            return nf.iloc[i + int(k)] / nf.iloc[i] - 1
        J["nifty"] = J.apply(hold, axis=1)
        J["hedged"] = J["net"] - J["nifty"] - cfg["hedge_cost_bps"] / 1e4
    daily = J.groupby("date")[[c for c in ("net", "excess", "net_stretch", "hedged") if c in J]].mean()
    k = cfg["ledger_top"]
    if "action" in J and J["action"].isin(["TRADE", "SKIP"]).any():
        daily["meta_filtered"] = (J["net"].where(J["action"] == "TRADE", 0.0)
                                  .groupby(J["date"]).sum() / k)
    if "size_x" in J and J["size_x"].notna().any():
        daily["meta_sized"] = (J["net"] * J["size_x"].fillna(1.0)).groupby(J["date"]).sum() / k
    if "meta_rank" in Jall and Jall["meta_rank"].notna().any():
        Jall["net"] = Jall["label_exit_ret"] - cost
        rr = Jall[Jall["meta_rank"] <= k]
        daily["meta_reranked"] = rr.groupby("date")["net"].sum() / k
    res["arms"] = {}
    for c in daily:
        x = daily[c].dropna().to_numpy()
        res["arms"][c] = (RC.block_bootstrap_mean(x) if len(x) >= cfg["min_days_for_ci"]
                          else {"mean": float(np.mean(x)) if len(x) else float("nan"),
                                "lo": float("nan"), "hi": float("nan"), "n_dates": len(x)})
    if verbose:
        names = {"net": "PRIMARY: top-3 net (ATR +1.5/-1.0)", "excess": "excess over buy-everything",
                 "net_stretch": "secondary: ATR +2.0 bracket", "hedged": "secondary: NIFTY-hedged",
                 "meta_filtered": "secondary: meta-filtered (per day)",
                 "meta_sized": "secondary: meta-sized (per day)",
                 "meta_reranked": "secondary: meta re-ranked (per day)"}
        print("\n" + "=" * 72)
        print(f"  FORWARD PAPER TEST - {res['resolved_days']} of {res['logged_days']} logged days "
              f"resolved | net of {cfg['cost_bps']:.0f} bp")
        print("=" * 72)
        for c, e in res["arms"].items():
            ci = f"[{e['lo']*1e4:+.0f},{e['hi']*1e4:+.0f}]" if np.isfinite(e["lo"]) else \
                f"(interval after {cfg['min_days_for_ci']} days)"
            print(f"  {names.get(c, c):<40} {e['mean']*1e4:+6.0f} bp {ci}")
        print("\n  Judge at the checkpoints fixed in advance (months 6 and 12), not on a good or bad week.")
    return res


# ----------------------------------------------------------------------
# HISTORY - how accurate the engine and the meta-model have been
# ----------------------------------------------------------------------
def _nifty_hold(root: Path, df: pd.DataFrame, date_col: str = "timestamp") -> pd.Series:
    """NIFTY 50 return over each trade's actual holding period (TP day, SL day or 5)."""
    from data_quality import _paths
    fp, _ = _paths(root, "NIFTY50")
    if not Path(fp).exists():
        return pd.Series(np.nan, index=df.index)
    nf = pd.read_parquet(fp, columns=["timestamp", "close"])
    nf["timestamp"] = _naive(nf["timestamp"]).dt.normalize()
    nf = nf.drop_duplicates("timestamp").set_index("timestamp")["close"].sort_index()
    pos = {t: i for i, t in enumerate(nf.index)}
    k = np.where(df["label_first_touch"] == 1, df["label_days_to_tp"],
                 np.where(df["label_first_touch"] == -1, df["label_days_to_sl"], 5.0))
    out = []
    for t, kk in zip(pd.to_datetime(df[date_col]).dt.normalize(), k):
        i = pos.get(t)
        out.append(np.nan if (i is None or not np.isfinite(kk) or i + int(kk) >= len(nf))
                   else nf.iloc[i + int(kk)] / nf.iloc[i] - 1)
    return pd.Series(out, index=df.index)


def _curve_stats(x: pd.Series) -> dict:
    x = x.dropna()
    if x.empty:
        return {}
    cum = x.cumsum()
    dd = cum - cum.cummax()
    neg = (x < 0).astype(int)
    streak = int((neg.groupby((neg != neg.shift()).cumsum()).cumsum() * neg).max())
    mon = x.groupby(x.index.to_period("M")).sum()
    return {"days": int(len(x)), "mean_bp": float(x.mean() * 1e4),
            "positive_days": float((x > 0).mean()), "positive_months": float((mon > 0).mean()),
            "total_bp": float(cum.iloc[-1] * 1e4), "max_drawdown_bp": float(dd.min() * 1e4),
            "longest_losing_streak_days": streak,
            "worst_month": (str(mon.idxmin()), float(mon.min() * 1e4)),
            "best_month": (str(mon.idxmax()), float(mon.max() * 1e4))}


def history(root: Path, verbose: bool = True) -> dict:
    out = _dir(root)
    vp = out / "validation_picks.parquet"
    if not vp.exists():
        raise SystemExit("no saved history - retrain with this version: python signal_engine.py train")
    E = load_engine(root)
    meta, cfg = E["meta"], E["meta"]["config"]
    cost = cfg["cost_bps"] / 1e4
    V = pd.read_parquet(vp)
    V["timestamp"] = _naive(V["timestamp"])
    U = pd.read_parquet(out / "validation_universe.parquet")["universe_net"]
    U.index = _naive(pd.Series(U.index)).to_numpy()
    T = V[V["rk"] <= cfg["ledger_top"]].copy()
    T["nifty"] = _nifty_hold(root, T)
    T["hedged"] = T["net"] - T["nifty"] - cfg["hedge_cost_bps"] / 1e4
    k = cfg["ledger_top"]
    B = pd.DataFrame({"net": T.groupby("timestamp")["net"].mean()})
    B["buy_everything"] = U.reindex(B.index).to_numpy()
    B["excess"] = B["net"] - B["buy_everything"]
    B["hedged"] = T.groupby("timestamp")["hedged"].mean()
    if STRETCH_LABEL in T:
        B["atr2_bracket"] = (T[STRETCH_LABEL] - cost).groupby(T["timestamp"]).mean()
    Mh = None
    if (out / "meta_history.parquet").exists():
        Mh = pd.read_parquet(out / "meta_history.parquet")
        Mh["timestamp"] = _naive(Mh["timestamp"])
        mt = Mh[Mh["rank"] <= k]
        B["meta_filtered"] = (mt["net"].where(mt["keep"].fillna(False).astype(bool), 0.0)
                              .groupby(mt["timestamp"]).sum() / k)
        if "w" in mt:
            B["meta_sized"] = (mt["net"] * mt["w"].fillna(1.0)).groupby(mt["timestamp"]).sum() / k
        if "mrank" in Mh:
            B["meta_reranked"] = Mh[Mh["mrank"] <= k].groupby("timestamp")["net"].sum() / k
    res = {"model_id": meta["model_id"], "period": f"{B.index.min().date()}..{B.index.max().date()}",
           "curves": {c: _curve_stats(B[c]) for c in B.columns if c != "buy_everything"},
           "buy_everything": _curve_stats(B["buy_everything"])}

    q = B.index.to_period("Q").astype(str)
    hitq = T.groupby(T["timestamp"].dt.to_period("Q").astype(str))["label_tp_before_sl"].mean()
    res["quarters"] = [{"quarter": qq, "days": int(len(g)), "net_bp": float(g["net"].mean() * 1e4),
                        "excess_bp": float(g["excess"].mean() * 1e4),
                        "market_bp": float(g["buy_everything"].mean() * 1e4),
                        "tp_first": float(hitq.get(qq, np.nan))} for qq, g in B.groupby(q)]

    def buckets(pred, real, n=5):
        ok_ = pred.notna() & real.notna()
        if ok_.sum() < n * 20:
            return []
        b = pd.qcut(pred[ok_].rank(method="first"), n, labels=False)
        return [{"bucket": int(i) + 1, "predicted": float(pred[ok_][b == i].mean()),
                 "realised": float(real[ok_][b == i].mean()), "n": int((b == i).sum())}
                for i in range(n)]
    acc = {"confidence": buckets(T["conf_cf"], T["label_tp_before_sl"], 10),
           "p_stop": buckets(T["p_sl"], (T["label_first_touch"] == -1).astype(float)),
           "mfe": buckets(T["mfe_hat"], T["label_mfe_5d"]),
           "mae": buckets(T["mae_hat"], T["label_mae_5d"])}
    tpf = T[T["label_first_touch"] == 1]
    acc["days"] = buckets(tpf["days_hat"], tpf["label_days_to_tp"], 4)
    acc["outcome_mix"] = {"tp_first": float((T["label_first_touch"] == 1).mean()),
                          "sl_first": float((T["label_first_touch"] == -1).mean()),
                          "timeout": float((T["label_first_touch"] == 0).mean()),
                          "mean_confidence": float(T["conf_cf"].mean())}
    res["accuracy"] = acc
    if Mh is not None:
        mt = Mh[Mh["rank"] <= k]
        res["meta"] = {"reliability": buckets(mt["meta_p"], mt["win"].astype(float), 10),
                       "days": int(mt["timestamp"].nunique()), "skipped": float(1 - mt["keep"].mean())}

    # trade log: every historical pick, prediction next to outcome
    TL = T.rename(columns={"timestamp": "date"})
    TL["outcome"] = TL["label_first_touch"].map({1: "TARGET", -1: "STOP", 0: "TIMEOUT"})
    TL["actual_days"] = np.where(TL["label_first_touch"] == 1, TL["label_days_to_tp"],
                                 np.where(TL["label_first_touch"] == -1, TL["label_days_to_sl"], 5))
    cols = ["date", "rk", "symbol", "conf_cf", "p_sl", "mfe_hat", "mae_hat", "days_hat", "outcome",
            "actual_days", "label_mfe_5d", "label_mae_5d", "net", "nifty", "hedged"]
    TL = TL[cols].rename(columns={"rk": "rank", "conf_cf": "pred_confidence", "p_sl": "pred_p_stop",
                                  "mfe_hat": "pred_mfe", "mae_hat": "pred_mae", "days_hat": "pred_days",
                                  "label_mfe_5d": "actual_mfe", "label_mae_5d": "actual_mae",
                                  "net": "net_return", "nifty": "nifty_same_period",
                                  "hedged": "hedged_return"})
    if Mh is not None:
        mcols = [c for c in ("meta_p", "keep", "w", "mrank") if c in Mh]
        TL = TL.merge(Mh[["timestamp", "symbol"] + mcols].rename(
            columns={"timestamp": "date", "meta_p": "meta_prob", "keep": "meta_keep",
                     "w": "meta_size", "mrank": "meta_rank"}), on=["date", "symbol"], how="left")
    TL.to_csv(out / "history_trades.csv", index=False)
    (out / "history_report.html").write_text(_history_html(res, B, meta), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    if verbose:
        _print_history(res, out)
    return res


def _print_history(res, out):
    print("\n" + "=" * 78)
    print(f"  HISTORY - engine {res['model_id']}, walk-forward (every prediction made before the day)")
    print(f"  {res['period']}")
    print("=" * 78)
    nm = {"net": "top-3 net", "excess": "excess over market", "hedged": "NIFTY-hedged",
          "atr2_bracket": "ATR +2.0 bracket", "meta_filtered": "meta-filtered",
          "meta_sized": "meta-sized", "meta_reranked": "meta re-ranked"}
    print(f"  {'book':<20}{'bp/day':>8}{'days+':>7}{'months+':>9}{'max DD bp':>11}{'losing run':>12}  worst month")
    for c, st in res["curves"].items():
        if st:
            print(f"  {nm.get(c, c):<20}{st['mean_bp']:>+8.1f}{st['positive_days']:>7.0%}"
                  f"{st['positive_months']:>9.0%}{st['max_drawdown_bp']:>+11.0f}"
                  f"{st['longest_losing_streak_days']:>9} d   {st['worst_month'][0]} {st['worst_month'][1]:+.0f}")
    a = res["accuracy"]
    print("\n  ACCURACY OF THE WATCHLIST NUMBERS (top-3 picks, unseen data)")
    for key, lab, fmt in (("confidence", "confidence", "{:.0%}"), ("p_stop", "P(stop)", "{:.0%}"),
                          ("mfe", "MFE", "{:+.1%}"), ("mae", "MAE", "{:+.1%}"), ("days", "days to TP", "{:.1f}")):
        if a.get(key):
            print(f"    {lab:<11} " + ", ".join(f"{fmt.format(r['predicted'])}->{fmt.format(r['realised'])}"
                                             for r in a[key]))
    om = a["outcome_mix"]
    print(f"    outcomes: target first {om['tp_first']:.0%} | stop first {om['sl_first']:.0%} | timeout "
          f"{om['timeout']:.0%} (mean stated confidence {om['mean_confidence']:.0%})")
    if res.get("meta") and res["meta"]["reliability"]:
        print("    meta       " + ", ".join(f"{r['predicted']:.0%}->{r['realised']:.0%}"
                                           for r in res["meta"]["reliability"]))
    print(f"\n  report: {out / 'history_report.html'}\n  trades: {out / 'history_trades.csv'}")


def _svg(B: pd.DataFrame, cols, labels, colors, w=960, h=280):
    C = B[cols].fillna(0).cumsum() * 1e4
    lo, hi = float(C.min().min()), float(C.max().max())
    hi = hi if hi > lo else lo + 1
    n = len(C)
    def xy(i, v):
        return 50 + (w - 70) * i / max(n - 1, 1), 15 + (h - 45) * (hi - v) / (hi - lo)
    parts = [f"<svg viewBox='0 0 {w} {h}' width='100%' role='img' aria-label='cumulative return'>"]
    z = xy(0, 0)[1]
    parts.append(f"<line x1='50' x2='{w-20}' y1='{z:.1f}' y2='{z:.1f}' stroke='currentColor' opacity='.25'/>")
    for c, col in zip(cols, colors):
        pts = " ".join(f"{xy(i, v)[0]:.1f},{xy(i, v)[1]:.1f}" for i, v in enumerate(C[c].to_numpy()))
        parts.append(f"<polyline fill='none' stroke='{col}' stroke-width='1.6' points='{pts}'/>")
    for t in (lo, 0, hi):
        parts.append(f"<text x='4' y='{xy(0, t)[1]+4:.1f}' font-size='11' fill='currentColor'>{t:+.0f}</text>")
    for frac in (0, .5, 1):
        i = int(frac * (n - 1))
        parts.append(f"<text x='{xy(i, lo)[0]-30:.1f}' y='{h-6}' font-size='11' fill='currentColor'>"
                     f"{C.index[i].date()}</text>")
    parts.append("</svg><p class='legend'>" + " ".join(
        f"<span style='color:{col}'>&#9632;</span> {lab}" for lab, col in zip(labels, colors)) + "</p>")
    return "".join(parts)


def _history_html(res, B, meta):
    import html as H
    css = ("body{font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1000px;margin:0 auto;"
           "padding:20px 14px 60px;color:#1d1d1f;background:#fff}@media(prefers-color-scheme:dark){body{"
           "color:#f2f2f7;background:#111113}}h1{font-size:22px}h2{font-size:17px;margin-top:30px}"
           ".mut{opacity:.7}.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;"
           "font-variant-numeric:tabular-nums}td,th{padding:5px 8px;border-bottom:1px solid rgba(128,128,128,.25);"
           "text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}.legend{font-size:13px}"
           ".box{border-left:3px solid #f59e0b;padding:8px 12px;background:rgba(245,158,11,.08)}")
    cols = [c for c in ("net", "excess", "hedged", "atr2_bracket", "meta_filtered", "meta_sized",
                        "meta_reranked") if c in B]
    labels = {"net": "top-3 net", "excess": "excess over market", "hedged": "NIFTY-hedged",
              "atr2_bracket": "ATR +2.0 bracket", "meta_filtered": "meta-filtered",
              "meta_sized": "meta-sized", "meta_reranked": "meta re-ranked"}
    colors = ["#2563eb", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#db2777", "#65a30d"]
    L = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
         f"content='width=device-width,initial-scale=1'><title>Engine history</title><style>{css}</style>"
         f"</head><body><h1>Signal engine history - {H.escape(res['model_id'])}</h1>"
         f"<p class='mut'>{res['period']} | walk-forward: every prediction was made by a model trained "
         f"only on earlier data | net of {meta['config']['cost_bps']:.0f} bp per trade</p>"
         "<div class='box'>This is the honest historical record. Scoring the past with the final model "
         "would look far better and mean nothing - it was trained on that past. The forward paper "
         "ledger (resolve) is the only evidence nobody has seen.</div>",
         "<h2>Cumulative return of the daily top-3 book (bp per slot, summed)</h2>",
         _svg(B, cols, [labels[c] for c in cols], colors[:len(cols)])]
    rows = "".join(
        f"<tr><td>{labels.get(c, c)}</td><td>{st['mean_bp']:+.1f}</td><td>{st['positive_days']:.0%}</td>"
        f"<td>{st['positive_months']:.0%}</td><td>{st['total_bp']:+.0f}</td><td>{st['max_drawdown_bp']:+.0f}</td>"
        f"<td>{st['longest_losing_streak_days']}</td><td>{st['worst_month'][0]} {st['worst_month'][1]:+.0f}</td></tr>"
        for c, st in res["curves"].items() if st)
    L.append("<div class='wrap'><table><tr><th>book</th><th>bp/day</th><th>days +</th><th>months +</th>"
             "<th>total bp</th><th>max drawdown</th><th>losing run (d)</th><th>worst month</th></tr>"
             + rows + "</table></div>")
    L.append("<h2>Quarter by quarter</h2><div class='wrap'><table><tr><th>quarter</th><th>days</th>"
             "<th>top-3 net</th><th>excess</th><th>market</th><th>target first</th></tr>" + "".join(
                 f"<tr><td>{r['quarter']}</td><td>{r['days']}</td><td>{r['net_bp']:+.0f}</td>"
                 f"<td>{r['excess_bp']:+.0f}</td><td>{r['market_bp']:+.0f}</td><td>{r['tp_first']:.0%}</td></tr>"
                 for r in res["quarters"]) + "</table></div>")
    a = res["accuracy"]
    L.append("<h2>Accuracy of every watchlist number (top-3 picks, unseen data)</h2>"
             "<p class='mut'>Picks sorted by what the engine predicted, then compared with what happened. "
             "Accurate means the two columns agree; useful means the realised column rises with the "
             "predicted one.</p>")
    for key, lab, f in (("confidence", "Confidence (P target first)", "{:.0%}"),
                        ("p_stop", "P(stop first)", "{:.0%}"), ("mfe", "Expected MFE", "{:+.2%}"),
                        ("mae", "Expected MAE", "{:+.2%}"), ("days", "Days to target (target-first picks)", "{:.2f}")):
        if a.get(key):
            L.append(f"<h3>{lab}</h3><div class='wrap'><table><tr><th>bucket</th><th>predicted</th>"
                     f"<th>realised</th><th>picks</th></tr>" + "".join(
                         f"<tr><td>{r['bucket']}</td><td>{f.format(r['predicted'])}</td>"
                         f"<td>{f.format(r['realised'])}</td><td>{r['n']}</td></tr>" for r in a[key])
                     + "</table></div>")
    om = a["outcome_mix"]
    L.append(f"<p>Top-3 outcomes: target first {om['tp_first']:.0%}, stop first {om['sl_first']:.0%}, "
             f"timeout {om['timeout']:.0%}; mean stated confidence {om['mean_confidence']:.0%}.</p>")
    if res.get("meta") and res["meta"]["reliability"]:
        L.append(f"<h2>Meta-model record</h2><p class='mut'>{res['meta']['days']} days tested walk-forward; "
                 f"skipped {res['meta']['skipped']:.0%} of picks.</p><div class='wrap'><table><tr>"
                 "<th>decile</th><th>stated P(profit)</th><th>realised</th><th>picks</th></tr>" + "".join(
                     f"<tr><td>{r['bucket']}</td><td>{r['predicted']:.0%}</td><td>{r['realised']:.0%}</td>"
                     f"<td>{r['n']}</td></tr>" for r in res["meta"]["reliability"]) + "</table></div>")
    L.append("<h2>Caveats</h2><p class='mut'>One flat cost; no slippage beyond it, no gaps through stops, "
             "no sizing or tax. The hedge uses NIFTY 50 over each trade's holding period plus 3 bp; the "
             "picks are mostly mid and small caps, so a real hedge may track less well.</p></body></html>")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["train", "watchlist", "resolve", "history"])
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT") or "")
    if not str(root):
        raise SystemExit("CACHE_DAILY_ROOT not set and --root not given")
    {"train": train, "watchlist": watchlist, "resolve": resolve, "history": history}[a.cmd](root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
