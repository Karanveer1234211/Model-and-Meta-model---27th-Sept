#!/usr/bin/env python3
"""
meta_label_test.py - can a second model learn WHEN to trust the ranking?

    python meta_label_test.py --root %CACHE_DAILY_ROOT%
    python meta_label_test.py --root %CACHE_DAILY_ROOT% --model M0_all_features

META-LABELING ON SAVED OUT-OF-SAMPLE PREDICTIONS
================================================
The primary model ranks stocks WITHIN a day, so anything identical for every
stock that day (market mood, breadth, volatility) cannot change who ranks
first. A trade / no-trade decision is not a ranking, and there those things
can matter. The meta-model looks only at the primary model's picks and asks:
is THIS pick likely to end in profit after costs?

Inputs per pick
  standing     rank, raw score, score versus the day's average (z), gap to the
               next-ranked stock
  market       canonical market state that day (research_common)
  the pick     ATR%, liquidity, 52-week position, drawdown
  model form   the primary top-3's mean net return over the previous 20 days,
               counting ONLY trades already finished by that day (entered at
               least 6 sessions earlier) - no peeking at open trades

  * learns from the primary model's OUT-OF-SAMPLE predictions only (saved by
    regime_research) - never from in-sample scores, which are overconfident
  * trains on the daily top-10 (more examples), acts only on the top-3
  * nested walk-forward: trained on earlier days, judged on later blocks, with
    a 5-session embargo (labels overlap)
  * shallow, heavily regularised model: ~18k examples with noisy outcomes

THE RULE, FIXED BEFORE IT RUNS
------------------------------
PRIMARY (filter): each day, skip any top-3 pick whose meta-probability is in
the least-promising 30%. The cut-off comes from the TRAINING years only (an
inner time split, so it is set on predictions the meta-model had not fitted).
A skipped pick's slot stays in cash. The filter joins the forward test only
if, day by day, filtered book minus plain top-3 book has a 95% block-bootstrap
interval ABOVE zero.
SECONDARY (reported only): size the top-3 by meta-probability.
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

CODE_VERSION = "meta_label_test v4"
CFG = {"cost_bps": 35.0, "train_top": 10, "act_top": 3, "skip_quantile": 0.30,
       "n_blocks": 4, "min_train_frac": 0.40, "embargo": 5, "inner_frac": 0.8,
       "form_windows": (5, 20, 60), "form_lag": 6, "bags": 10, "bag_frac": 0.8, "seed": 7,
       "oof_blocks": 5, "oof_bags": 3, "rerank_pool": 5,
       "hgb": {"max_iter": 150, "max_depth": 3, "learning_rate": 0.05,
               "min_samples_leaf": 100, "l2_regularization": 2.0}}
PICK_COLS = {"D_atr_pct": "pick_atr_pct", "X_turnover_med": "pick_turnover",
             "D_pos_in_52w_range": "pick_52w_pos", "D_drawdown_252": "pick_drawdown"}


def _log(msg):
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def _naive(s):
    t = pd.to_datetime(s)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    return t


def load_predictions(root: Path, run: str | None, model: str, cfg: dict):
    """Every stock's out-of-sample prediction for one model of a regime_research run."""
    base = root / "panel" / "research"
    runs = sorted(p for p in base.glob("RUN_*") if (p / "p6_pred").exists())
    rd = base / run if run else runs[-1]
    files = sorted((rd / "p6_pred").glob(f"{model}_fold*.parquet"))
    if not files:
        raise SystemExit(f"no saved predictions for {model} in {rd}")
    pr = pd.concat([pd.read_parquet(f, columns=["timestamp", "symbol", "p_raw", "exit_ret"])
                    for f in files], ignore_index=True)
    pr["timestamp"] = _naive(pr["timestamp"])
    pr = pr.dropna(subset=["p_raw", "exit_ret"])
    pr["net"] = pr["exit_ret"] - cfg["cost_bps"] / 1e4
    return rd, pr


def build_dataset(root: Path, run: str | None, model: str, cfg: dict):
    rd, pr = load_predictions(root, run, model, cfg)
    D, feats, _ = meta_features(pr, panel_extras(root), cfg)
    return rd, D, feats


def _clf(cfg, seed=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(early_stopping=False,
                                          random_state=cfg["seed"] if seed is None else seed,
                                          **cfg["hgb"])


# ======================================================================
# SHARED META CORE (used by this test and by signal_engine)
# ======================================================================
HELPER_COLS = ("p_sl", "mfe_hat", "mae_hat", "days_hat")
PICK_EXTRA = ("pick_ret1", "pick_ret5", "pick_gapvol20")
SOUL_META = ("soul_p_tp", "soul_p_ret", "soul_p_tp_vs_market", "soul_p_ret_vs_market", "soul_p_neff",
             "soul_m_tp", "soul_m_ret", "soul_m_gap_ret", "soul_m_tp_se")
MARKET_EXTRA = ("INDIAVIX_close", "INDIAVIX_ret_1d", "MKT_D_dist_from_52wh", "MKT_D_realvol_20",
                "MKT_D_adx14", "MKT_D_rsi14")


def meta_features(pr: pd.DataFrame, extras: pd.DataFrame | None, cfg: dict,
                  form_series: pd.Series | None = None):
    """
    pr: EVERY stock scored on each day (timestamp, symbol, p_raw, and net / helper
    columns when known). Returns (top-N rows with meta features, feature list,
    daily top-k net series used for model form).
    """
    pr = pr.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    g = pr.groupby("timestamp")["p_raw"]
    pr["rank"] = g.rank(ascending=False, method="first")
    pr["score_z"] = (pr["p_raw"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    pr = pr.sort_values(["timestamp", "rank"]).reset_index(drop=True)
    pr["gap_next"] = pr["p_raw"] - pr.groupby("timestamp")["p_raw"].shift(-1)
    s3 = pr[pr["rank"] == 3].set_index("timestamp")["p_raw"]
    s4 = pr[pr["rank"] == 4].set_index("timestamp")["p_raw"]
    pr["day_gap_3_4"] = pr["timestamp"].map(s3 - s4)
    # crowding at the top: how many stocks sit within 1% (relative) of the #3 score
    thr = pr["timestamp"].map(s3 * 0.99)
    pr["top_crowding"] = (pr["p_raw"] >= thr).groupby(pr["timestamp"]).transform("sum")
    # signal persistence: days in the top-10 over the PREVIOUS 5 sessions
    top10 = (pr["rank"] <= 10).astype(float)
    piv = pd.DataFrame({"t": pr["timestamp"], "s": pr["symbol"], "v": top10}).pivot_table(
        index="t", columns="s", values="v", aggfunc="max").fillna(0.0)
    streak = piv.shift(1).rolling(5, min_periods=1).sum().stack()
    streak.index.names = ["timestamp", "symbol"]
    pr = pr.merge(streak.rename("top10_streak5").reset_index(), on=["timestamp", "symbol"], how="left")
    # model form: realised top-k net over past windows, ONLY finished trades
    if form_series is None and "net" in pr:
        form_series = (pr[pr["rank"] <= cfg["act_top"]].groupby("timestamp")["net"].mean().sort_index())
    fcols = []
    if form_series is not None and len(form_series):
        lagged = form_series.shift(cfg["form_lag"])
        for w in cfg["form_windows"]:
            f = lagged.rolling(w, min_periods=max(3, w // 2)).mean()
            pr[f"model_form_{w}"] = pr["timestamp"].map(f)
            fcols.append(f"model_form_{w}")
    top = pr[pr["rank"] <= cfg["train_top"]].copy()
    if extras is not None:
        top = top.merge(extras, on=["timestamp", "symbol"], how="left")
    feats = ["rank", "p_raw", "score_z", "gap_next", "day_gap_3_4", "top_crowding", "top10_streak5"] + fcols
    feats += [c for c in RC.MARKET_FEATURES if c in top]
    feats += [c for c in MARKET_EXTRA if c in top]
    feats += [v for v in PICK_COLS.values() if v in top]
    feats += [c for c in HELPER_COLS if c in top]
    feats += [c for c in PICK_EXTRA if c in top]
    feats += [c for c in SOUL_META if c in top and top[c].notna().any()]
    if "net" in top:
        top["win"] = (top["net"] > 0).astype(int)
    return top.sort_values(["timestamp", "rank"]).reset_index(drop=True), feats, form_series


def panel_extras(root: Path) -> pd.DataFrame:
    """Per (timestamp, symbol): canonical market state, VIX / NIFTY context, pick traits."""
    import pyarrow.parquet as pq
    pp = root / "panel" / "panel.parquet"
    have = set(pq.ParquetFile(pp).schema_arrow.names)
    pc = [c for c in PICK_COLS if c in have]
    mx = [c for c in MARKET_EXTRA if c in have]
    oc = ["open"] if "open" in have else []
    P = pd.read_parquet(pp, columns=["timestamp", "symbol", "close"] + oc + pc + mx)
    P["timestamp"] = _naive(P["timestamp"])
    P = P.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    sessions = np.sort(P["timestamp"].unique())
    dcode = np.searchsorted(sessions, P["timestamp"].to_numpy())
    seg = RC.session_segments(dcode, pd.factorize(P["symbol"])[0])
    mk = pd.DataFrame(RC.market_state(pd.to_numeric(P["close"], errors="coerce").to_numpy(),
                                      dcode, seg, len(sessions)))
    mk["timestamp"] = sessions
    cl = pd.Series(pd.to_numeric(P["close"], errors="coerce").to_numpy())
    prev = cl.groupby(seg).shift(1)
    P["pick_ret1"] = (cl / prev - 1).to_numpy()
    P["pick_ret5"] = (cl / cl.groupby(seg).shift(5) - 1).to_numpy()
    if oc:
        gap = pd.Series(pd.to_numeric(P["open"], errors="coerce").to_numpy()) / prev - 1
        P["pick_gapvol20"] = gap.groupby(seg).transform(
            lambda x: x.rolling(20, min_periods=10).std()).to_numpy()
    px = [c for c in PICK_EXTRA if c in P]
    out = P[["timestamp", "symbol"] + pc + mx + px].rename(columns=PICK_COLS).merge(mk, on="timestamp", how="left")
    sf = pp.parent / "soul" / "soul_v3.parquet"
    if sf.exists():
        import pyarrow.parquet as pq2
        sc = [c for c in SOUL_META if c in pq2.ParquetFile(sf).schema_arrow.names]
        if sc:
            S = pd.read_parquet(sf, columns=["timestamp", "symbol"] + sc)
            S["timestamp"] = _naive(S["timestamp"]).astype("datetime64[ns]")
            out["timestamp"] = out["timestamp"].astype("datetime64[ns]")
            out = pd.merge_asof(out.sort_values("timestamp"), S.sort_values("timestamp"),
                                on="timestamp", by="symbol", direction="backward")
    if "pick_turnover" in out:
        out["pick_turnover"] = np.log10(pd.to_numeric(out["pick_turnover"], errors="coerce").clip(lower=1))
    return out


class MetaModel:
    """
    Bagged, calibrated meta-classifier.

    B shallow boosted models, each on a random 80% of training DAYS (whole days,
    so overlapping labels stay together), averaged -> less noise than one model.
    Isotonic calibration is fitted on an inner time split the members had not
    seen, so a stated 60% means about 60% on new data. The spread across members
    is the per-pick uncertainty.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, D: pd.DataFrame, feats: list):
        """
        Calibration from OUT-OF-FOLD predictions across ALL training days: the
        days are cut into contiguous blocks; each block is predicted by members
        trained on the other blocks (with a buffer so overlapping labels never
        straddle). v2 calibrated on one late slice and drifted.
        """
        from sklearn.isotonic import IsotonicRegression
        cfg = self.cfg
        self.feats = list(feats)
        days = np.sort(D["timestamp"].unique())
        oof = np.full(len(D), np.nan)
        buf = pd.Timedelta(days=10)
        for blk in np.array_split(days, cfg["oof_blocks"]):
            lo, hi = pd.Timestamp(blk[0]), pd.Timestamp(blk[-1])
            inb = (D["timestamp"] >= lo) & (D["timestamp"] <= hi)
            tr = D[~((D["timestamp"] >= lo - buf) & (D["timestamp"] <= hi + buf))]
            if tr["win"].nunique() < 2 or not inb.any():
                continue
            mem = self._bag(tr, bags=cfg["oof_bags"])
            X = D.loc[inb, self.feats].to_numpy("float32")
            oof[inb.to_numpy()] = np.mean([m.predict_proba(X)[:, 1] for m in mem], axis=0)
        ok = np.isfinite(oof)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
            oof[ok], D["win"].to_numpy()[ok])
        top = ok & (D["rank"].to_numpy() <= cfg["act_top"])
        self.tau = float(np.quantile(self.iso.predict(oof[top]), cfg["skip_quantile"])) if top.any() else 0.0
        self.members = self._bag(D)
        return self

    def _bag(self, D, bags=None):
        cfg = self.cfg
        rng = np.random.default_rng(cfg["seed"])
        days = np.sort(D["timestamp"].unique())
        out = []
        for b in range(bags or cfg["bags"]):
            pick = set(rng.choice(days, int(len(days) * cfg["bag_frac"]), replace=False))
            sub = D[D["timestamp"].isin(pick)]
            out.append(_clf(cfg, seed=cfg["seed"] + b).fit(sub[self.feats].to_numpy("float32"), sub["win"]))
        return out

    def _raw(self, D):
        X = D[self.feats].to_numpy("float32")
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.members])

    def predict(self, D):
        R = self._raw(D)
        cal = np.column_stack([self.iso.predict(R[:, b]) for b in range(R.shape[1])])
        return (self.iso.predict(R.mean(axis=1)), np.percentile(cal, 10, axis=1),
                np.percentile(cal, 90, axis=1))


def nested_evaluation(D: pd.DataFrame, feats: list, cfg: dict, return_frame: bool = False):
    """Walk-forward over days: meta fitted on earlier days, judged on later blocks."""
    from sklearn.metrics import roc_auc_score
    days = np.sort(D["timestamp"].unique())
    edges = np.linspace(int(len(days) * cfg["min_train_frac"]), len(days), cfg["n_blocks"] + 1).astype(int)
    parts, blocks = [], []
    for b in range(cfg["n_blocks"]):
        te0, te1 = edges[b], edges[b + 1]
        tr_end = days[te0 - cfg["embargo"] - 1]
        tr = D[D["timestamp"] <= tr_end]
        te = D[(D["timestamp"] >= days[te0]) & (D["timestamp"] <= days[te1 - 1])].copy()
        mm = MetaModel(cfg).fit(tr, feats)
        te["meta_p"], te["meta_lo"], te["meta_hi"] = mm.predict(te)
        te["tau"], te["block"] = mm.tau, b + 1
        parts.append(te)
        blocks.append({"block": b + 1, "train_through": str(pd.Timestamp(tr_end).date()),
                       "test": f"{pd.Timestamp(days[te0]).date()}..{pd.Timestamp(days[te1-1]).date()}",
                       "cutoff": mm.tau})
    T = pd.concat(parts, ignore_index=True)
    k = cfg["act_top"]
    A = T[T["rank"] <= k].copy()
    A["keep"] = A["meta_p"] >= A["tau"]
    base = A.groupby("timestamp")["net"].sum() / k
    filt = A[A["keep"]].groupby("timestamp")["net"].sum().reindex(base.index, fill_value=0.0) / k
    A["w"] = (A["meta_p"] / A.groupby("timestamp")["meta_p"].transform("mean")).clip(0.5, 1.5)
    A["w"] = A["w"] * k / A.groupby("timestamp")["w"].transform("sum")
    size = (A["w"] * A["net"]).groupby(A["timestamp"]).sum() / k
    # RE-RANK: the best k by meta-probability from the engine's top-N - every slot invested
    R5 = T[T["rank"] <= cfg["rerank_pool"]].copy()
    R5["mrank"] = R5.groupby("timestamp")["meta_p"].rank(ascending=False, method="first")
    rer = (R5[R5["mrank"] <= k].groupby("timestamp")["net"].sum() / k).reindex(base.index)
    y, p = A["win"].to_numpy(), A["meta_p"].to_numpy()
    base_rate = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    brier_ref = float(np.mean((base_rate - y) ** 2))
    A["dec"] = pd.qcut(A["meta_p"].rank(method="first"), 10, labels=False) + 1
    res = {"days_tested": int(base.shape[0]), "picks_tested": int(len(A)),
           "skipped_share": float(1 - A["keep"].mean()),
           "days_all_cash": int((A.groupby("timestamp")["keep"].sum() == 0).sum()),
           "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
           "brier": brier, "brier_skill": float(1 - brier / brier_ref) if brier_ref > 0 else float("nan"),
           "base_rate": base_rate,
           "band_coverage": float(((A["meta_lo"] <= A["meta_p"]) & (A["meta_p"] <= A["meta_hi"])).mean()),
           "reliability": [{"decile": int(d), "predicted": float(g["meta_p"].mean()),
                            "realised": float(g["win"].mean()), "net_bp": float(g["net"].mean() * 1e4),
                            "picks": int(len(g))} for d, g in A.groupby("dec")],
           "baseline": RC.block_bootstrap_mean(base.to_numpy(), seed=cfg["seed"]),
           "filtered": RC.block_bootstrap_mean(filt.to_numpy(), seed=cfg["seed"]),
           "sized": RC.block_bootstrap_mean(size.to_numpy(), seed=cfg["seed"]),
           "filter_minus_base": RC.block_bootstrap_mean((filt - base).to_numpy(), seed=cfg["seed"]),
           "size_minus_base": RC.block_bootstrap_mean((size - base).to_numpy(), seed=cfg["seed"]),
           "reranked": RC.block_bootstrap_mean(rer.dropna().to_numpy(), seed=cfg["seed"]),
           "rerank_minus_base": RC.block_bootstrap_mean((rer - base).dropna().to_numpy(), seed=cfg["seed"]),
           "blocks": blocks}
    # WITHIN-RANK LIFT: among picks of the SAME rank, did higher meta earn more?
    # This is the meta-model's OWN contribution. The decile spread also contains
    # the main model's ranking, because rank is one of meta's inputs.
    lift = {}
    for r_, g in A.groupby("rank"):
        med = g["meta_p"].median()
        hi_, lo_ = g[g["meta_p"] > med]["net"], g[g["meta_p"] <= med]["net"]
        if len(hi_) > 20 and len(lo_) > 20:
            lift[int(r_)] = float((hi_.mean() - lo_.mean()) * 1e4)
    res["within_rank_lift_bp"] = lift
    res["within_rank_lift_mean_bp"] = float(np.mean(list(lift.values()))) if lift else float("nan")
    res["by_block"] = []
    for b, g in A.groupby("block"):
        bb = g.groupby("timestamp")["net"].sum() / k
        ff = g[g["keep"]].groupby("timestamp")["net"].sum().reindex(bb.index, fill_value=0.0) / k
        res["by_block"].append({"block": int(b), "base_bp": float(bb.mean() * 1e4),
                                "filtered_bp": float(ff.mean() * 1e4), "skipped": float(1 - g["keep"].mean())})
    dm, ds = res["filter_minus_base"], res["size_minus_base"]
    res["filter_validated"] = bool(np.isfinite(dm["lo"]) and dm["lo"] > 0)
    res["sizing_validated"] = bool(np.isfinite(ds["lo"]) and ds["lo"] > 0)
    dr = res["rerank_minus_base"]
    res["rerank_validated"] = bool(np.isfinite(dr["lo"]) and dr["lo"] > 0)
    res["joins_forward_test"] = res["filter_validated"]
    if return_frame:
        F = R5.merge(A[["timestamp", "symbol", "keep", "w"]], on=["timestamp", "symbol"], how="left")
        return res, F[["timestamp", "symbol", "rank", "mrank", "meta_p", "meta_lo", "meta_hi", "tau",
                       "keep", "w", "block", "net", "win"]].copy()
    return res


def leans_on(mm: "MetaModel", top: int = 8):
    g = np.zeros(len(mm.feats))
    try:
        for m in mm.members:
            for it in m._predictors:
                nd = it[0].nodes
                sp = nd["is_leaf"] == 0
                np.add.at(g, nd["feature_idx"][sp], nd["gain"][sp])
    except Exception:
        return []
    if g.sum() <= 0:
        return []
    return [(mm.feats[i], float(g[i] / g.sum())) for i in np.argsort(-g)[:top]]


def run(root: Path, run: str | None = None, model: str = "M0_all_features",
        overrides: dict | None = None, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    cfg = {**CFG, **(overrides or {})}
    rd, D, feats = build_dataset(root, run, model, cfg)
    _log(f"{rd.name} / {model}: {D['timestamp'].nunique():,} out-of-sample days, {len(D):,} "
         f"top-{cfg['train_top']} picks, {len(feats)} meta features, {cfg['bags']}-model bag")
    res = nested_evaluation(D, feats, cfg)
    res.update({"code": CODE_VERSION, "run": rd.name, "model": model, "cost_bps": cfg["cost_bps"],
                "built_at": dt.datetime.now().isoformat(), "features": feats})
    res["leans_on"] = leans_on(MetaModel(cfg).fit(D, feats))
    res["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    (rd / "meta_label.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    RC.ledger_append(root / "panel" / "panel.parquet",
                     {"kind": "exploration", "tool": CODE_VERSION, "model": model,
                      "joins_forward_test": res["joins_forward_test"]})
    if verbose:
        print_meta(res)
    return res


def _f(e):
    return f"{e['mean']*1e4:+5.1f} [{e['lo']*1e4:+5.1f},{e['hi']*1e4:+5.1f}]"


def print_meta(res, title: str | None = None):
    print("\n" + "=" * 78)
    print("  " + (title or f"META-MODEL - {res.get('model', '')} ({res.get('run', '')}), unseen picks, "
                            f"net of {res.get('cost_bps', 35):.0f} bp"))
    print("=" * 78)
    print(f"  tested {res['days_tested']} days, {res['picks_tested']} top-3 picks | skipped "
          f"{res['skipped_share']:.0%}, {res['days_all_cash']} days fully in cash")
    print("\n  HOW PREDICTIVE (unseen data)")
    print(f"    AUC {res['auc']:.3f}   (0.500 = coin flip)")
    print(f"    Brier skill {res['brier_skill']:+.3f}   (above 0 = better than always guessing the "
          f"base rate of {res['base_rate']:.0%})")
    print("\n  HOW ACCURATE - when it says X%, how often did picks win? (deciles, unseen)")
    print("    " + ", ".join(f"{r['predicted']:.0%}->{r['realised']:.0%}" for r in res["reliability"]))
    print("    net bp by decile: " + " ".join(f"{r['net_bp']:+.0f}" for r in res["reliability"]))
    if res.get("within_rank_lift_bp"):
        print("\n  META'S OWN CONTRIBUTION - same rank, higher-meta half minus lower-meta half (net bp/pick)")
        print("    " + " | ".join(f"rank {r}: {v:+.0f}" for r, v in res["within_rank_lift_bp"].items())
              + "   (the decile spread above also contains the main model's ranking)")
    print("\n  DAILY BOOK RETURN, bp per day (3 slots; a skipped slot earns 0)")
    print(f"    plain top-3        {_f(res['baseline'])}")
    print(f"    meta-filtered      {_f(res['filtered'])}")
    print(f"    meta-sized         {_f(res['sized'])}   (secondary)")
    if "reranked" in res:
        print(f"    meta-reranked      {_f(res['reranked'])}   (secondary: best 3 of the engine's top-5)")
    print("  by test block: " + " | ".join(f"B{b['block']} plain {b['base_bp']:+.0f} / filtered "
                                          f"{b['filtered_bp']:+.0f}" for b in res["by_block"]))
    if res.get("leans_on"):
        print("\n  LEANS ON: " + ", ".join(f"{f} {w:.0%}" for f, w in res["leans_on"][:6]))
    print("\n" + "-" * 78)
    print("  RULE: filtered minus plain, day by day, 95% interval above zero")
    print(f"  filtered minus plain {_f(res['filter_minus_base'])} bp/day | sized minus plain "
          f"{_f(res['size_minus_base'])}"
          + (f" | reranked minus plain {_f(res['rerank_minus_base'])}" if "rerank_minus_base" in res else ""))
    print("  RESULT: " + ("VALIDATED - the filter joins the forward test" if res["filter_validated"]
                          else "NOT validated - it does not clearly improve the book")
          + ("; sizing validated" if res["sizing_validated"] else "")
          + ("; re-ranking validated" if res.get("rerank_validated") else ""))
    print("-" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--model", default="M0_all_features")
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT") or "")
    if not str(root):
        raise SystemExit("CACHE_DAILY_ROOT not set and --root not given")
    run(root, a.run, a.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
