#!/usr/bin/env python3
"""
challenger_test.py - two open questions, one run, identical folds.

    pip install lightgbm xgboost catboost          (once)
    python challenger_test.py --root %CACHE_DAILY_ROOT%

EXPERIMENT 1 - DOES MACRO / NIFTY HELP RANKING?
    A          engine clean features, HistGradientBoosting (the engine's model)
    A-macro    the same without market / macro inputs: NIFTY features (MKT_*),
               stock-versus-market features (X_rel*), and the outside series'
               daily returns (sector indices, VIX, USD/INR, crude, gold ...)
  NIFTY and macro values are identical for every stock on a day, so they can
  only matter to a within-day ranking through interactions. This measures it.
  RULE: macro HELPS if A minus A-macro (daily top-3 net) has a 95% interval
  above zero; it is REMOVED only if A-macro minus A is above zero; anything
  in between = no clear effect, nothing changes.

EXPERIMENT 2 - IS ANOTHER BOOSTER BETTER?
    HGB (current), LightGBM, XGBoost, CatBoost - matched capacity, default-ish
    settings, NOT tuned per library - and ENSEMBLE = the average of their
    per-day rank percentiles.
  RULE: the ENSEMBLE replaces HGB only if ENSEMBLE minus HGB has a 95%
  interval above zero. Single libraries are reported for INFORMATION only:
  picking the best of four after the fact would reward luck.

  * identical folds, training rows (sampled once per fold), target
    (label_tp_before_sl) and RAW-score ranking for every model
  * all labelled sessions (the lockbox is spent); walk-forward, expanding,
    5-session embargo; net of 35 bp; block-bootstrap intervals over days
  * a library that is not installed is skipped and reported; the ensemble
    then averages the models that ran
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

CODE_VERSION = "challenger_test v1"
CFG = {"cost_bps": 35.0, "n_splits": 5, "min_train_frac": 0.35, "embargo": 5,
       "fit_cap": 400_000, "top_n": (1, 3, 5, 10), "seed": 7,
       "hgb": {"max_iter": 300, "max_depth": 5, "learning_rate": 0.06,
               "min_samples_leaf": 200, "l2_regularization": 1.0},
       "lgbm": {"n_estimators": 300, "learning_rate": 0.06, "num_leaves": 31, "max_depth": 5,
                "min_child_samples": 200, "reg_lambda": 1.0, "verbose": -1, "n_jobs": -1},
       "xgb": {"n_estimators": 300, "learning_rate": 0.06, "max_depth": 5, "min_child_weight": 40,
               "reg_lambda": 1.0, "tree_method": "hist", "n_jobs": -1, "eval_metric": "logloss"},
       "cat": {"iterations": 300, "learning_rate": 0.06, "depth": 5, "l2_leaf_reg": 1.0,
               "verbose": 0, "thread_count": -1}}
NAMES = {"HGB": "HistGradientBoosting (current)", "HGB_NOMACRO": "HGB without macro",
         "LGBM": "LightGBM", "XGB": "XGBoost", "CAT": "CatBoost", "ENSEMBLE": "ensemble (rank average)"}


def _log(msg):
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def macro_columns(feats):
    """Market / macro inputs: NIFTY features, stock-vs-market, outside-series returns."""
    out = []
    for c in feats:
        if c.startswith("MKT_") or c.startswith("X_rel"):
            out.append(c)
        elif c.endswith("_ret_1d") and not c.startswith(("D_", "X_", "R_")):
            pref = c[: -len("_ret_1d")]
            if pref and pref.upper() == pref:
                out.append(c)
    return out


def _builders(cfg):
    from sklearn.ensemble import HistGradientBoostingClassifier
    b = {"HGB": lambda: HistGradientBoostingClassifier(early_stopping=False, random_state=cfg["seed"],
                                                        **cfg["hgb"])}
    missing = []
    try:
        import lightgbm as lgb
        b["LGBM"] = lambda: lgb.LGBMClassifier(random_state=cfg["seed"], **cfg["lgbm"])
    except Exception:
        missing.append("lightgbm")
    try:
        import xgboost as xgb
        b["XGB"] = lambda: xgb.XGBClassifier(random_state=cfg["seed"], **cfg["xgb"])
    except Exception:
        missing.append("xgboost")
    try:
        import catboost as cb
        b["CAT"] = lambda: cb.CatBoostClassifier(random_seed=cfg["seed"], **cfg["cat"])
    except Exception:
        missing.append("catboost")
    return b, missing


def decide(diff: dict, better: str, worse: str) -> str:
    """'better'/'worse'/'no clear effect' from a bootstrap dict of (A minus B)."""
    if np.isfinite(diff["lo"]) and diff["lo"] > 0:
        return better
    if np.isfinite(diff["hi"]) and diff["hi"] < 0:
        return worse
    return "no clear effect"


def run(root: Path, overrides: dict | None = None, verbose: bool = True) -> dict:
    import feasibility_test as FT
    import signal_engine as SE
    t0 = time.perf_counter()
    cfg = {**CFG, **(overrides or {})}
    pp = root / "panel" / "panel.parquet"
    feats, excluded = SE.clean_features(pp)
    macro = macro_columns(feats)
    nomacro = [c for c in feats if c not in set(macro)]
    builders, missing = _builders(cfg)
    p = pd.read_parquet(pp, columns=["timestamp", "symbol", "label_tp_before_sl", "label_exit_ret"] + feats)
    p["timestamp"] = SE._naive(p["timestamp"])
    p = p.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    ok = (p["label_tp_before_sl"].notna() & p["label_exit_ret"].notna()).to_numpy()
    p = p[ok].reset_index(drop=True)
    ts = p["timestamp"].to_numpy()
    y = p["label_tp_before_sl"].to_numpy().astype(int)
    er = p["label_exit_ret"].to_numpy("float64")
    sessions = np.sort(np.unique(ts))
    _log(f"{len(p):,} rows | {len(feats)} features, of which {len(macro)} macro/market | models: "
         f"{', '.join(builders)}" + (f" | NOT installed: {', '.join(missing)}" if missing else ""))
    folds_def = FT.splits(sessions, cfg["n_splits"], cfg["embargo"], cfg["min_train_frac"])
    rng = np.random.default_rng(cfg["seed"])
    ci_nomacro = [feats.index(c) for c in nomacro]
    preds = {k: [] for k in list(builders) + ["HGB_NOMACRO", "ENSEMBLE"]}
    folds = []
    for s in folds_def:
        tf = time.perf_counter()
        tr = np.where(ts <= s["train_end"])[0]
        te = np.where((ts >= s["test_start"]) & (ts <= s["test_end"]))[0]
        if len(tr) > cfg["fit_cap"]:
            tr = np.sort(rng.choice(tr, cfg["fit_cap"], replace=False))   # SAME rows for every model
        Xtr = p[feats].iloc[tr].to_numpy("float32")
        Xte = p[feats].iloc[te].to_numpy("float32")
        scores = {}
        for nm, mk in builders.items():
            m = mk().fit(Xtr, y[tr])
            scores[nm] = m.predict_proba(Xte)[:, 1]
        m = builders["HGB"]().fit(Xtr[:, ci_nomacro], y[tr])
        scores["HGB_NOMACRO"] = m.predict_proba(Xte[:, ci_nomacro])[:, 1]
        del Xtr, Xte
        day = pd.Series(ts[te])
        ranks = [pd.Series(scores[k]).groupby(day).rank(pct=True).to_numpy() for k in builders]
        scores["ENSEMBLE"] = np.mean(ranks, axis=0)
        row = {"fold": s["fold"], "test": f"{pd.Timestamp(s['test_start']).date()}.."
                                          f"{pd.Timestamp(s['test_end']).date()}"}
        for nm, sc in scores.items():
            d = pd.DataFrame({"timestamp": ts[te], "p_raw": sc, "exit_ret": er[te],
                              "y": (er[te] - cfg["cost_bps"] / 1e4 > 0).astype(float)})
            preds[nm].append(d)
            e3 = RC.topn_evidence(d, 3, cost_bps=cfg["cost_bps"], B=300, seed=cfg["seed"], score_col="p_raw")
            row[f"{nm}_top3"] = e3["net"]["mean"]
        folds.append(row)
        _log(f"fold {s['fold']} {row['test']}: top-3 net " + " | ".join(
            f"{nm} {row[nm + '_top3']*1e4:+.0f}" for nm in scores) + f" ({time.perf_counter()-tf:.0f}s)")

    res = {"code": CODE_VERSION, "built_at": dt.datetime.now().isoformat(), "cost_bps": cfg["cost_bps"],
           "macro_columns": macro, "missing_libraries": missing, "folds": folds, "models": {}}
    net_day, uni = {}, None
    for nm, parts in preds.items():
        d = pd.concat(parts, ignore_index=True)
        d["net"] = d["exit_ret"] - cfg["cost_bps"] / 1e4
        if uni is None:
            uni = d.groupby("timestamp")["net"].mean()
            res["universe"] = RC.block_bootstrap_mean(uni.to_numpy(), seed=cfg["seed"])
        t3 = RC.daily_topn_values(d, 3, "net", "p_raw")
        net_day[nm] = t3
        res["models"][nm] = {
            "rank_ic": FT.rank_ic_by_date(d["timestamp"], d["p_raw"], d["exit_ret"]),
            "topn": {int(n): RC.topn_evidence(d, n, cost_bps=cfg["cost_bps"], seed=cfg["seed"],
                                              score_col="p_raw") for n in cfg["top_n"]},
            "excess3": RC.block_bootstrap_mean((t3 - uni.reindex(t3.index)).dropna().to_numpy(), seed=cfg["seed"])}

    def diff(a, b):
        x = (net_day[a] - net_day[b].reindex(net_day[a].index)).dropna()
        return RC.block_bootstrap_mean(x.to_numpy(), seed=cfg["seed"])
    res["macro_diff"] = diff("HGB", "HGB_NOMACRO")          # A minus A-macro
    res["macro_verdict"] = decide(res["macro_diff"], "macro HELPS ranking - keep it",
                                  "macro HURTS ranking - remove it")
    res["ensemble_diff"] = diff("ENSEMBLE", "HGB")
    res["ensemble_verdict"] = ("ENSEMBLE replaces HGB" if decide(res["ensemble_diff"], "b", "w") == "b"
                               else "HGB stays")
    res["singles_vs_hgb"] = {nm: diff(nm, "HGB") for nm in ("LGBM", "XGB", "CAT") if nm in net_day}
    res["folds_won"] = {k: int(sum(f[f"{k}_top3"] > f["HGB_top3"] for f in folds))
                        for k in ("ENSEMBLE", "LGBM", "XGB", "CAT") if f"{k}_top3" in folds[0]}
    res["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    out = root / "panel" / "engine"
    out.mkdir(parents=True, exist_ok=True)
    (out / "challenger.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    RC.ledger_append(pp, {"kind": "exploration", "tool": CODE_VERSION, "macro": res["macro_verdict"],
                          "ensemble": res["ensemble_verdict"]})
    if verbose:
        _print(res)
    return res


def _f(e):
    return f"{e['mean']*1e4:+5.0f} [{e['lo']*1e4:+4.0f},{e['hi']*1e4:+4.0f}]"


def _print(res):
    print("\n" + "=" * 80)
    print(f"  CHALLENGER TEST - identical folds and rows, net of {res['cost_bps']:.0f} bp "
          f"({res['minutes']} min)")
    print("=" * 80)
    print(f"  buy everything: {_f(res['universe'])} bp per trade")
    print(f"\n  {'model':<32}{'rank IC':>9}{'top-3 net [95%]':>20}{'top-3 excess':>18}")
    for nm, r in res["models"].items():
        print(f"  {NAMES.get(nm, nm):<32}{r['rank_ic']:>+9.4f}{_f(r['topn'][3]['net']):>20}"
              f"{_f(r['excess3']):>18}")
    print("\n  per fold, top-3 net bp:")
    for f in res["folds"]:
        print(f"    F{f['fold']} {f['test']}: " + " | ".join(
            f"{k.replace('_top3', '')} {v*1e4:+.0f}" for k, v in f.items() if k.endswith("_top3")))
    print("\n" + "-" * 80)
    print(f"  1. MACRO: A minus A-without-macro {_f(res['macro_diff'])} bp/day")
    print(f"     removed for this test: {len(res['macro_columns'])} columns, e.g. "
          f"{', '.join(res['macro_columns'][:6])}{' ...' if len(res['macro_columns']) > 6 else ''}")
    print(f"     VERDICT: {res['macro_verdict']}")
    print(f"  2. ALGORITHM: ensemble minus HGB {_f(res['ensemble_diff'])} bp/day | folds won "
          f"{res['folds_won'].get('ENSEMBLE', '-')}/{len(res['folds'])}")
    print(f"     VERDICT: {res['ensemble_verdict']}")
    if res["singles_vs_hgb"]:
        print("     for information only (best-of-four is not a valid choice): " + " | ".join(
            f"{NAMES[k]} minus HGB {_f(v)}" for k, v in res["singles_vs_hgb"].items()))
    if res["missing_libraries"]:
        print(f"     not installed, skipped: {', '.join(res['missing_libraries'])} "
              f"(pip install {' '.join(res['missing_libraries'])})")
    print("-" * 80)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT") or "")
    if not str(root):
        raise SystemExit("CACHE_DAILY_ROOT not set and --root not given")
    run(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
