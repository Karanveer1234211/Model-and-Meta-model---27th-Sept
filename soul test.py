#!/usr/bin/env python3
"""
soul_test.py - does the Soul add anything the engine does not already know?

    python soul_test.py --root %CACHE_DAILY_ROOT%

THE QUESTION
============
Soul's 22 features are analogue forecasts, built strictly from the past: for
each stock and day, what followed SIMILAR earlier moments - in that stock's
own history ("self": its personality) and across all stocks ("cross") - plus
where the two disagree and how strong the evidence is. The main model never
knows which stock it is looking at, so "this stock tends to follow through"
is information it does not have. Whether that helps is an empirical question.

    A  engine features (signal_engine.clean_features)
    B  engine features + the 22 Soul features

  * identical rows (only rows where Soul exists), folds, settings, target
    (label_tp_before_sl) and RAW-score ranking
  * Soul is joined with a strictly BACKWARD as-of match: each row sees the
    most recent Soul value computed on or before its own date. Soul's own
    construction only uses episodes whose outcome had resolved (query date
    minus 5 trading sessions)
  * all labelled sessions from Soul's first date (the lockbox is spent);
    walk-forward, expanding, 5-session embargo

THE RULE, FIXED BEFORE IT RUNS
------------------------------
Soul joins the engine only if, day by day, B's top-3 net return minus A's has
a 95% block-bootstrap interval ABOVE zero. Folds won are reported as support.
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

CODE_VERSION = "soul_test v1"
CFG = {"cost_bps": 35.0, "n_splits": 5, "min_train_frac": 0.35, "embargo": 5,
       "fit_cap": 400_000, "top_n": (1, 3, 5, 10), "seed": 7,
       "hgb": {"max_iter": 300, "max_depth": 5, "learning_rate": 0.06,
               "min_samples_leaf": 200, "l2_regularization": 1.0}}


def _log(msg):
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def _ns(s):
    t = pd.to_datetime(s)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    return t.astype("datetime64[ns]")


def load(root: Path, cfg: dict):
    import signal_engine as SE
    pp = root / "panel" / "panel.parquet"
    sd = root / "panel" / "soul"
    sp = Path(cfg["soul_file"]) if cfg.get("soul_file") else (
        sd / "soul_v3.parquet" if (sd / "soul_v3.parquet").exists() else sd / "soul_features.parquet")
    if not sp.exists():
        raise SystemExit(f"no Soul features at {sp} - build them first (see soul_features.py)")
    feats, excluded = SE.clean_features(pp)
    p = pd.read_parquet(pp, columns=["timestamp", "symbol", "label_tp_before_sl", "label_exit_ret"] + feats)
    p["timestamp"] = _ns(p["timestamp"])
    f = pd.read_parquet(sp)
    f["timestamp"] = _ns(f["timestamp"])
    soul = [c for c in f.columns if c.startswith("soul_")]
    RC.assert_no_label_leak(soul, "soul_test")
    p = pd.merge_asof(p.sort_values("timestamp"), f[["timestamp", "symbol"] + soul].sort_values("timestamp"),
                      on="timestamp", by="symbol", direction="backward")
    p = p.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    first = f["timestamp"].min()
    ok = (p["timestamp"] >= first) & p[soul].notna().any(axis=1) & \
        p["label_tp_before_sl"].notna() & p["label_exit_ret"].notna()
    return p[ok].reset_index(drop=True), feats, soul, {"soul_file": sp.name, "soul_first": str(first.date()),
                                                       "soul_last": str(f["timestamp"].max().date()),
                                                       "excluded": excluded}


def _clf(cfg):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(early_stopping=False, random_state=cfg["seed"], **cfg["hgb"])


def _gain_share(model, names, subset):
    g = np.zeros(len(names))
    try:
        for it in model._predictors:
            nd = it[0].nodes
            sp = nd["is_leaf"] == 0
            np.add.at(g, nd["feature_idx"][sp], nd["gain"][sp])
    except Exception:
        return None, []
    if g.sum() <= 0:
        return None, []
    idx = [i for i, n in enumerate(names) if n in subset]
    top = sorted(((names[i], g[i] / g.sum()) for i in idx), key=lambda x: -x[1])[:6]
    return float(g[idx].sum() / g.sum()), top


def run(root: Path, overrides: dict | None = None, verbose: bool = True) -> dict:
    import feasibility_test as FT
    t0 = time.perf_counter()
    cfg = {**CFG, **(overrides or {})}
    p, feats, soul, info = load(root, cfg)
    ts = p["timestamp"].to_numpy()
    y = p["label_tp_before_sl"].to_numpy().astype(int)
    er = p["label_exit_ret"].to_numpy("float64")
    sessions = np.sort(np.unique(ts))
    _log(f"{len(p):,} rows with Soul | {len(feats)} engine features + {len(soul)} Soul features | "
         f"{pd.Timestamp(sessions[0]).date()} -> {pd.Timestamp(sessions[-1]).date()}")
    folds_def = FT.splits(sessions, cfg["n_splits"], cfg["embargo"], cfg["min_train_frac"])
    rng = np.random.default_rng(cfg["seed"])
    cols = {"A": feats, "B": feats + soul}
    preds = {"A": [], "B": []}
    folds, shares = [], []
    for s in folds_def:
        tf = time.perf_counter()
        tr = np.where(ts <= s["train_end"])[0]
        te = np.where((ts >= s["test_start"]) & (ts <= s["test_end"]))[0]
        if len(tr) > cfg["fit_cap"]:
            tr = np.sort(rng.choice(tr, cfg["fit_cap"], replace=False))   # SAME rows for A and B
        row = {"fold": s["fold"], "test": f"{pd.Timestamp(s['test_start']).date()}.."
                                          f"{pd.Timestamp(s['test_end']).date()}"}
        for nm in ("A", "B"):
            m = _clf(cfg).fit(p[cols[nm]].iloc[tr].to_numpy("float32"), y[tr])
            sc = m.predict_proba(p[cols[nm]].iloc[te].to_numpy("float32"))[:, 1]
            d = pd.DataFrame({"timestamp": ts[te], "p_raw": sc, "exit_ret": er[te],
                              "y": (er[te] - cfg["cost_bps"] / 1e4 > 0).astype(float)})
            preds[nm].append(d)
            e3 = RC.topn_evidence(d, 3, cost_bps=cfg["cost_bps"], B=300, seed=cfg["seed"], score_col="p_raw")
            row[f"{nm}_top3_net"] = e3["net"]["mean"]
            if nm == "B":
                sh, top = _gain_share(m, cols["B"], set(soul))
                shares.append({"fold": s["fold"], "soul_share": sh, "top": top})
        folds.append(row)
        _log(f"fold {s['fold']} {row['test']}: top-3 net A {row['A_top3_net']*1e4:+.0f}bp / "
             f"B (with Soul) {row['B_top3_net']*1e4:+.0f}bp ({time.perf_counter()-tf:.0f}s)")

    res = {"code": CODE_VERSION, "built_at": dt.datetime.now().isoformat(), "cost_bps": cfg["cost_bps"],
           **info, "n_soul": len(soul), "folds": folds, "soul_gain": shares, "pooled": {}}
    net_day, uni = {}, None
    for nm, parts in preds.items():
        d = pd.concat(parts, ignore_index=True)
        d["net"] = d["exit_ret"] - cfg["cost_bps"] / 1e4
        if uni is None:
            uni = d.groupby("timestamp")["net"].mean()
            res["universe"] = RC.block_bootstrap_mean(uni.to_numpy(), seed=cfg["seed"])
        import feasibility_test as FT2
        ic = FT2.rank_ic_by_date(d["timestamp"], d["p_raw"], d["exit_ret"])
        ev = {int(n): RC.topn_evidence(d, n, cost_bps=cfg["cost_bps"], seed=cfg["seed"], score_col="p_raw")
              for n in cfg["top_n"]}
        t3 = RC.daily_topn_values(d, 3, "net", "p_raw")
        net_day[nm] = t3
        res["pooled"][nm] = {"rank_ic": ic, "topn": ev,
                             "excess3": RC.block_bootstrap_mean((t3 - uni.reindex(t3.index)).dropna().to_numpy(),
                                                                seed=cfg["seed"])}
    diff = (net_day["B"] - net_day["A"].reindex(net_day["B"].index)).dropna()
    res["b_minus_a"] = RC.block_bootstrap_mean(diff.to_numpy(), seed=cfg["seed"])
    res["folds_won"] = int(sum(f["B_top3_net"] > f["A_top3_net"] for f in folds))
    dm = res["b_minus_a"]
    res["joins_engine"] = bool(np.isfinite(dm["lo"]) and dm["lo"] > 0)
    res["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    out = root / "panel" / "soul"
    (out / "soul_test.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    RC.ledger_append(root / "panel" / "panel.parquet",
                     {"kind": "exploration", "tool": CODE_VERSION, "joins_engine": res["joins_engine"]})
    if verbose:
        _print(res)
    return res


def _f(e):
    return f"{e['mean']*1e4:+5.0f} [{e['lo']*1e4:+4.0f},{e['hi']*1e4:+4.0f}]"


def _print(res):
    print("\n" + "=" * 78)
    print(f"  ENGINE vs ENGINE + SOUL ({res.get('soul_file','')}) - net of {res['cost_bps']:.0f} bp | {res['soul_first']} -> "
          f"{res['soul_last']}")
    print("=" * 78)
    print(f"  buy everything: {_f(res['universe'])} bp per trade")
    print(f"\n  {'':<16}{'rank IC':>9}" + "".join(f"{'top-'+str(n)+' net':>20}" for n in (1, 3, 5, 10)))
    for nm, lab in (("A", "A: engine"), ("B", "B: engine+Soul")):
        r = res["pooled"][nm]
        print(f"  {lab:<16}{r['rank_ic']:>+9.4f}" + "".join(f"{_f(r['topn'][n]['net']):>20}" for n in (1, 3, 5, 10)))
    print(f"\n  excess over buy-everything, top-3: A {_f(res['pooled']['A']['excess3'])} | "
          f"B {_f(res['pooled']['B']['excess3'])}")
    print("  per fold, top-3 net bp: " + " | ".join(
        f"F{f['fold']} A {f['A_top3_net']*1e4:+.0f} / B {f['B_top3_net']*1e4:+.0f}" for f in res["folds"]))
    sh = [s["soul_share"] for s in res["soul_gain"] if s["soul_share"] is not None]
    if sh:
        last = res["soul_gain"][-1]
        print(f"\n  how much model B leaned on Soul: {np.mean(sh):.0%} of split gain on average "
              f"(Soul is {res['n_soul']} of its inputs)")
        if last["top"]:
            print("  top Soul inputs (last fold): " + ", ".join(f"{n} {w:.1%}" for n, w in last["top"]))
    print("\n" + "-" * 78)
    print("  RULE: B top-3 minus A top-3, day by day, 95% interval above zero")
    print(f"  B minus A: {_f(res['b_minus_a'])} bp | folds won {res['folds_won']}/{len(res['folds'])}")
    print("  RESULT: " + ("SOUL JOINS the engine" if res["joins_engine"] else
                          "Soul does NOT join - no clear improvement over the engine alone"))
    print("-" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--soul-file", default=None, help="default: soul_v3.parquet if built, else soul_features.parquet")
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT") or "")
    if not str(root):
        raise SystemExit("CACHE_DAILY_ROOT not set and --root not given")
    run(root, overrides={"soul_file": a.soul_file} if a.soul_file else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
