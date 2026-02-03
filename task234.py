# -*- coding: utf-8 -*-
"""
Task234++ (Season-level) — Task2/3/4 pipeline built on the "stable" Bayesian vote-share estimator.

Per season:
1) Fit Bayesian weekly fan vote shares from judge scores + elimination info (via first.py).
2) Task2: Compare elimination under "rank" vs "percent" vote rules; agreement, bias, robustness.
3) Task3: Analyze drivers of fan vs judge preference using a simple pooled regression + correlations.
4) Task4++: Propose & replay a new hybrid voting rule; grid-search alpha schedules / protections.

Outputs: outputs_task234/seasonX/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from first import LowDimBayesVoteEstimator


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def rank_desc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(-x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)

    uniq = {}
    for i, v in enumerate(x):
        uniq.setdefault(v, []).append(i)
    for v, idxs in uniq.items():
        if len(idxs) > 1:
            ranks[idxs] = float(np.mean(ranks[idxs]))
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    rx = rank_desc(x)
    ry = rank_desc(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def make_estimator(data_path: str, random_seed: int):
    try:
        return LowDimBayesVoteEstimator(data_path=data_path, random_seed=random_seed)
    except TypeError:
        return LowDimBayesVoteEstimator(random_seed=random_seed)


# -----------------------------
# Task2 — compare voting rules
# -----------------------------
def compute_elim_by_rule(week_idx: np.ndarray, rule_scores: np.ndarray) -> int:
    local_elim = int(np.argmin(rule_scores))
    return int(week_idx[local_elim])


def plot_task2_summary(df_task2: pd.DataFrame, out_dir: str, season_num: int) -> None:
    ensure_dir(out_dir)

    agreement = float(df_task2["rank_equals_percent"].mean()) if len(df_task2) else float("nan")

    def _acc(col: str) -> float:
        x = df_task2[col].replace("", np.nan).astype(float)
        return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")

    rank_acc = _acc("rank_matches_actual")
    percent_acc = _acc("percent_matches_actual")

    plt.figure(figsize=(8, 5))
    plt.bar(["Agreement", "Rank-Acc", "Percent-Acc"], [agreement, rank_acc, percent_acc])
    plt.ylim(0, 1)
    plt.title(f"Task2 Summary (Season {season_num})")
    plt.ylabel("Rate")
    path = os.path.join(out_dir, f"season{season_num}_task2_rates.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_task2_fanbias(
    df_task2: pd.DataFrame,
    estimator,
    votes: Dict[int, Dict[str, Dict[str, float]]],
    season_num: int,
    out_dir: str,
) -> None:
    ensure_dir(out_dir)
    sd = estimator.preprocess_data(season_num)
    weeks = sd["weeks"]
    celebs = sd["celebrities"]
    alive = sd["alive_mask"]

    weeks_list = [int(x) for x in list(weeks)]

    rank_method_ranks = []
    percent_method_ranks = []

    name_to_idx = {c: i for i, c in enumerate(celebs)}

    for _, r in df_task2.iterrows():
        wk = int(r["week"])
        if wk not in weeks_list:
            continue
        t = int(weeks_list.index(wk))

        idx = np.where(alive[t])[0]
        if wk not in votes:
            continue

        V_all = np.array([votes[wk][c]["mean"] for c in celebs], dtype=float)
        V = V_all[idx]
        if np.sum(V) <= 0:
            continue
        rV = rank_desc(V / np.sum(V))  # 1 = most fan-loved

        elim_rank = r["elim_rank_method"]
        elim_percent = r["elim_percent_method"]

        if elim_rank in name_to_idx and name_to_idx[elim_rank] in idx:
            elim_rank_local = int(np.where(idx == name_to_idx[elim_rank])[0][0])
            rank_method_ranks.append(float(rV[elim_rank_local]))
        if elim_percent in name_to_idx and name_to_idx[elim_percent] in idx:
            elim_percent_local = int(np.where(idx == name_to_idx[elim_percent])[0][0])
            percent_method_ranks.append(float(rV[elim_percent_local]))

    if not rank_method_ranks or not percent_method_ranks:
        return

    plt.figure(figsize=(10, 4))
    plt.boxplot([rank_method_ranks, percent_method_ranks], labels=["Rank method", "Percent method"])
    plt.title(f"Fan-bias (rank of eliminated in fan votes) — Season {season_num}")
    plt.ylabel("Fan-rank of eliminated (1=best, larger=worse)")
    path = os.path.join(out_dir, f"season{season_num}_task2_fanbias.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def task2_compare_methods(
    estimator,
    season_num: int,
    votes: Dict[int, Dict[str, Dict[str, float]]],
    out_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    sd = estimator.preprocess_data(season_num)
    weeks = sd["weeks"]
    celebs = sd["celebrities"]
    J = sd["J"]
    alive = sd["alive_mask"]

    elim_weeks_idx = sd["elim_weeks_idx"]
    elim_obs = sd["elim_obs"]
    actual_elim = {int(weeks[int(elim_weeks_idx[k])]): int(elim_obs[k]) for k in range(len(elim_weeks_idx))}

    rows = []
    controversies = []

    for t, wk in enumerate(weeks):
        wk = int(wk)
        if wk not in votes:
            continue

        idx = np.where(alive[t])[0]
        if len(idx) <= 1:
            continue

        V_all = np.array([votes[wk][c]["mean"] for c in celebs], dtype=float)
        V = V_all[idx]

        Jraw_all = J[t].astype(float)
        Jraw = Jraw_all[idx]
        if np.sum(Jraw) <= 0:
            continue

        rJ = rank_desc(Jraw)
        rV = rank_desc(V)
        total_rank = rJ + rV
        elim_rank_local = int(np.argmax(total_rank))
        elim_rank = int(idx[elim_rank_local])

        PJ = Jraw / float(np.sum(Jraw))
        PV = V / float(np.sum(V))
        S_percent = 0.5 * PJ + 0.5 * PV
        elim_percent = compute_elim_by_rule(idx, S_percent)

        actual = actual_elim.get(wk, None)

        qcut = max(1, int(np.ceil(0.25 * len(idx))))
        rJ_full = rank_desc(Jraw)
        elim_rank_topJ = int(rJ_full[elim_rank_local] <= qcut)

        elim_percent_local = int(np.where(idx == elim_percent)[0][0])
        elim_percent_topJ = int(rJ_full[elim_percent_local] <= qcut)

        rows.append({
            "season": season_num,
            "week": wk,
            "actual_elim": str(celebs[actual]) if actual is not None else "",
            "elim_rank_method": str(celebs[elim_rank]),
            "elim_percent_method": str(celebs[elim_percent]),
            "rank_matches_actual": int(elim_rank == actual) if actual is not None else "",
            "percent_matches_actual": int(elim_percent == actual) if actual is not None else "",
            "rank_equals_percent": int(elim_rank == elim_percent),
        })

        controversies.append({
            "season": season_num,
            "week": wk,
            "rank_elim": str(celebs[elim_rank]),
            "rank_elim_in_judge_top_quartile": elim_rank_topJ,
            "percent_elim": str(celebs[elim_percent]),
            "percent_elim_in_judge_top_quartile": elim_percent_topJ,
        })

    df = pd.DataFrame(rows)
    df_cont = pd.DataFrame(controversies)

    df.to_csv(os.path.join(out_dir, f"season{season_num}_task2_compare.csv"), index=False)
    df_cont.to_csv(os.path.join(out_dir, f"season{season_num}_task2_controversies.csv"), index=False)

    plot_task2_summary(df, out_dir, season_num)
    plot_task2_fanbias(df, estimator, votes, season_num, out_dir)

    return df, df_cont


def task2_fliprate_under_noise(
    estimator,
    season_num: int,
    votes: Dict[int, Dict[str, Dict[str, float]]],
    out_dir: str,
    eps_list: List[float] = [0.02, 0.05, 0.10],
    n_mc: int = 250,
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)

    sd = estimator.preprocess_data(season_num)
    weeks = sd["weeks"]
    celebs = sd["celebrities"]
    J = sd["J"]
    alive = sd["alive_mask"]

    rows_rank = []
    rows_percent = []

    for t, wk in enumerate(weeks):
        wk = int(wk)
        if wk not in votes:
            continue

        idx = np.where(alive[t])[0]
        if len(idx) <= 1:
            continue

        V_all = np.array([votes[wk][c]["mean"] for c in celebs], dtype=float)
        V0 = V_all[idx]
        if np.sum(V0) <= 0:
            continue
        V0 = V0 / np.sum(V0)

        Jraw = J[t].astype(float)[idx]
        if np.sum(Jraw) <= 0:
            continue
        PJ = Jraw / np.sum(Jraw)

        rJ0 = rank_desc(Jraw)
        rV0 = rank_desc(V0)
        base_rank_local = int(np.argmax(rJ0 + rV0))
        base_rank = int(idx[base_rank_local])

        base_percent = compute_elim_by_rule(idx, 0.5 * PJ + 0.5 * V0)

        for eps in eps_list:
            flips_rank = 0
            flips_percent = 0
            for _ in range(n_mc):
                noise = rng.normal(0.0, eps, size=len(idx))
                Vp = np.clip(V0 * (1.0 + noise), 1e-9, None)
                Vp = Vp / np.sum(Vp)

                rVp = rank_desc(Vp)
                elim_rank_local = int(np.argmax(rJ0 + rVp))
                elim_rank = int(idx[elim_rank_local])
                if elim_rank != base_rank:
                    flips_rank += 1

                elim_percent = compute_elim_by_rule(idx, 0.5 * PJ + 0.5 * Vp)
                if elim_percent != base_percent:
                    flips_percent += 1

            rows_rank.append({"season": season_num, "week": wk, "eps": eps, "flip_rate": flips_rank / n_mc})
            rows_percent.append({"season": season_num, "week": wk, "eps": eps, "flip_rate": flips_percent / n_mc})

    df_rank = pd.DataFrame(rows_rank)
    df_percent = pd.DataFrame(rows_percent)

    df_rank.to_csv(os.path.join(out_dir, f"season{season_num}_task2_fliprate_rank.csv"), index=False)
    df_percent.to_csv(os.path.join(out_dir, f"season{season_num}_task2_fliprate_percent.csv"), index=False)
    plot_task2_fliprate(df_rank, df_percent, out_dir, season_num)

    return df_rank, df_percent


def plot_task2_fliprate(df_rank: pd.DataFrame, df_percent: pd.DataFrame, out_dir: str, season_num: int) -> None:
    ensure_dir(out_dir)
    if df_rank.empty or df_percent.empty:
        return

    rank_avg = df_rank.groupby("eps")["flip_rate"].mean()
    percent_avg = df_percent.groupby("eps")["flip_rate"].mean()
    eps_list = sorted(rank_avg.index.tolist())

    plt.figure(figsize=(10, 5))
    x = np.arange(len(eps_list))
    w = 0.35
    plt.bar(x - w / 2, [rank_avg[e] for e in eps_list], width=w, label="Rank method")
    plt.bar(x + w / 2, [percent_avg[e] for e in eps_list], width=w, label="Percent method")
    plt.xticks(x, [f"eps={e:.2f}" for e in eps_list])
    plt.ylim(0, 1)
    plt.title("Task2 Robustness: elimination flip rate under vote perturbation")
    plt.ylabel("Flip rate (higher = less robust)")
    plt.legend()
    path = os.path.join(out_dir, f"season{season_num}_task2_fliprate.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------
# Task3 — drivers / factors
# -----------------------------
def plot_task3_scatter(df_panel: pd.DataFrame, out_dir: str, season_num: int) -> None:
    ensure_dir(out_dir)
    if df_panel.empty:
        return

    g = df_panel.groupby("celeb").agg(
        fan_share=("fan_share", "mean"),
        judge_z=("judge_z", "mean"),
    ).reset_index()

    plt.figure(figsize=(8, 6))
    plt.scatter(g["judge_z"], g["fan_share"], s=80)
    plt.title(f"Task3: Judge vs Fan Preference (Season {season_num})")
    plt.xlabel("Judge preference (mean weekly z-score)")
    plt.ylabel("Fan preference (mean vote share)")
    path = os.path.join(out_dir, f"season{season_num}_task3_judge_vs_fan.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def task3_build_panel(
    estimator,
    season_num: int,
    votes: Dict[int, Dict[str, Dict[str, float]]],
    out_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    sd = estimator.preprocess_data(season_num)
    weeks = sd["weeks"]
    celebs = sd["celebrities"]
    J = sd["J"]
    alive = sd["alive_mask"]

    rows = []
    for t, wk in enumerate(weeks):
        wk = int(wk)
        if wk not in votes:
            continue

        idx = np.where(alive[t])[0]
        if len(idx) <= 1:
            continue

        V_all = np.array([votes[wk][c]["mean"] for c in celebs], dtype=float)
        V = V_all[idx]
        V = V / float(np.sum(V))

        Jraw = J[t].astype(float)[idx]
        mu = float(np.mean(Jraw))
        sdv = float(np.std(Jraw) + 1e-9)
        Jz = (Jraw - mu) / sdv

        for j, g in enumerate(idx):
            rows.append({
                "season": season_num,
                "week": wk,
                "celeb": celebs[int(g)],
                "fan_share": float(V[j]),
                "judge_score": float(Jraw[j]),
                "judge_z": float(Jz[j]),
            })

    df_panel = pd.DataFrame(rows)
    df_panel.to_csv(os.path.join(out_dir, f"season{season_num}_task3_panel.csv"), index=False)

    X = np.column_stack([np.ones(len(df_panel)), df_panel["judge_z"].to_numpy(dtype=float)])
    y = df_panel["fan_share"].to_numpy(dtype=float)

    XtX = X.T @ X
    try:
        beta = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.array([float("nan"), float("nan")])

    yhat = X @ beta
    ssr = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2) + 1e-12)
    r2 = 1.0 - ssr / sst

    pear = float(np.corrcoef(df_panel["judge_z"].to_numpy(), df_panel["fan_share"].to_numpy())[0, 1]) if len(df_panel) > 2 else float("nan")
    spear = spearman_corr(df_panel["judge_z"].to_numpy(), df_panel["fan_share"].to_numpy())

    df_coef = pd.DataFrame([{
        "season": season_num,
        "beta_intercept": float(beta[0]),
        "beta_judge_z": float(beta[1]),
        "R2": float(r2),
        "PearsonCorr(judge_z, fan_share)": pear,
        "SpearmanCorr(judge_z, fan_share)": spear,
        "n_rows": int(len(df_panel)),
    }])
    df_coef.to_csv(os.path.join(out_dir, f"season{season_num}_task3_coef_compare.csv"), index=False)

    plot_task3_scatter(df_panel, out_dir, season_num)
    return df_panel, df_coef


# -----------------------------
# Task4++ — new rule simulation + tuning
# -----------------------------
@dataclass(frozen=True)
class NewRuleConfig:
    name: str
    alpha_mode: str
    alpha_start: float = 0.65
    alpha_end: float = 0.35
    alpha_fixed: float = 0.50
    protect_top_quartile: bool = True


def task4_simulate_new_rule(
    estimator,
    season_num: int,
    votes: dict,
    cfg: NewRuleConfig,
) -> pd.DataFrame:
    sd = estimator.preprocess_data(season_num)
    weeks = sd["weeks"]
    celebs = sd["celebrities"]
    J = sd["J"]
    alive = sd["alive_mask"]

    elim_weeks_idx = sd["elim_weeks_idx"]
    elim_obs = sd["elim_obs"]
    actual_elim = {int(weeks[int(elim_weeks_idx[k])]): int(elim_obs[k]) for k in range(len(elim_weeks_idx))}

    rows: List[Dict[str, Any]] = []
    T = len(weeks)

    for t, wk in enumerate(weeks):
        wk = int(wk)
        if wk not in votes:
            continue

        idx = np.where(alive[t])[0]
        if len(idx) <= 1:
            continue

        V_all = np.array([votes[wk][c]["mean"] for c in celebs], dtype=float)
        V = V_all[idx]
        if np.sum(V) <= 0:
            continue
        PV = V / float(np.sum(V))

        Jraw = J[t].astype(float)[idx]
        if np.sum(Jraw) <= 0:
            continue
        PJ = Jraw / float(np.sum(Jraw))

        if cfg.alpha_mode == "fixed":
            alpha = float(cfg.alpha_fixed)
        else:
            u = (t + 1) / max(1, T)
            alpha = float(cfg.alpha_start * (1.0 - u) + cfg.alpha_end * u)

        S = alpha * PJ + (1.0 - alpha) * PV

        qcut = max(1, int(np.ceil(0.25 * len(idx))))
        rJ = rank_desc(Jraw)
        protected = (rJ <= qcut) if cfg.protect_top_quartile else np.zeros(len(idx), dtype=bool)

        cand = np.where(~protected)[0]
        if len(cand) == 0:
            cand = np.arange(len(idx))

        pred_local = int(cand[np.argmin(S[cand])])
        pred = int(idx[pred_local])

        actual = actual_elim.get(wk, None)
        matches = (pred == actual) if actual is not None else np.nan
        pred_topJ = int(rJ[pred_local] <= qcut)

        rows.append({
            "season": season_num,
            "week": wk,
            "rule_name": cfg.name,
            "alpha_week": float(alpha),
            "protect_top_quartile": int(cfg.protect_top_quartile),
            "actual_elim": str(celebs[actual]) if actual is not None else "",
            "pred_newrule_elim": str(celebs[pred]),
            "newrule_matches_actual": int(matches) if matches == matches else "",
            "pred_elim_in_judge_top_quartile": pred_topJ,
        })

    return pd.DataFrame(rows)


def plot_task4_summary(df_sum: pd.DataFrame, out_dir: str, season_num: int, best_rule_name: str) -> None:
    ensure_dir(out_dir)
    plt.figure(figsize=(10, 5))

    if df_sum.empty:
        plt.title(f"Task4 Summary (Season {season_num}) — no data")
        plt.savefig(os.path.join(out_dir, f"season{season_num}_task4_summary.png"), dpi=200)
        plt.close()
        return

    topk = df_sum.head(6).copy()
    x = np.arange(len(topk))
    w = 0.35
    plt.bar(x - w / 2, topk["acc"].to_numpy(), width=w, label="Accuracy")
    plt.bar(x + w / 2, (1.0 - topk["controversial_rate"].to_numpy()), width=w, label="(1 - Controversy)")

    plt.xticks(x, [str(n).replace(" ", "") for n in topk["rule_name"].tolist()], rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.title(f"Task4 Summary (Season {season_num}) — best: {best_rule_name}")
    plt.legend()
    path = os.path.join(out_dir, f"season{season_num}_task4_summary.png")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def task4_grid_search(
    estimator,
    season_num: int,
    votes: dict,
    out_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)

    configs: List[NewRuleConfig] = [
        NewRuleConfig(name="A(0.70→0.30)+Prot", alpha_mode="adaptive", alpha_start=0.70, alpha_end=0.30, protect_top_quartile=True),
        NewRuleConfig(name="A(0.65→0.35)+Prot", alpha_mode="adaptive", alpha_start=0.65, alpha_end=0.35, protect_top_quartile=True),
        NewRuleConfig(name="A(0.60→0.40)+Prot", alpha_mode="adaptive", alpha_start=0.60, alpha_end=0.40, protect_top_quartile=True),
        NewRuleConfig(name="A(0.65→0.35)+NoProt", alpha_mode="adaptive", alpha_start=0.65, alpha_end=0.35, protect_top_quartile=False),
        NewRuleConfig(name="F(0.55)+Prot", alpha_mode="fixed", alpha_fixed=0.55, protect_top_quartile=True),
        NewRuleConfig(name="F(0.50)+Prot", alpha_mode="fixed", alpha_fixed=0.50, protect_top_quartile=True),
        NewRuleConfig(name="F(0.60)+Prot", alpha_mode="fixed", alpha_fixed=0.60, protect_top_quartile=True),
        NewRuleConfig(name="F(0.55)+NoProt", alpha_mode="fixed", alpha_fixed=0.55, protect_top_quartile=False),
    ]

    all_replays = []
    summary_rows = []

    for cfg in configs:
        df_rep = task4_simulate_new_rule(estimator, season_num, votes, cfg)
        if df_rep.empty:
            continue

        all_replays.append(df_rep)

        x = df_rep["newrule_matches_actual"].replace("", np.nan).astype(float)
        acc = float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")
        cr = float(df_rep["pred_elim_in_judge_top_quartile"].astype(float).mean())
        score = acc - 0.25 * cr

        summary_rows.append({
            "season": season_num,
            "rule_name": cfg.name,
            "acc": acc,
            "controversial_rate": cr,
            "score": score,
            "alpha_mode": cfg.alpha_mode,
            "alpha_start": cfg.alpha_start,
            "alpha_end": cfg.alpha_end,
            "alpha_fixed": cfg.alpha_fixed,
            "protect_top_quartile": int(cfg.protect_top_quartile),
            "n_weeks_evaluated": int(np.isfinite(x).sum()),
        })

    df_sum = pd.DataFrame(summary_rows).sort_values(["score", "acc"], ascending=False).reset_index(drop=True)
    df_sum.to_csv(os.path.join(out_dir, f"season{season_num}_task4_gridsearch_summary.csv"), index=False)

    df_all = pd.concat(all_replays, ignore_index=True) if all_replays else pd.DataFrame()
    df_all.to_csv(os.path.join(out_dir, f"season{season_num}_task4_allrules_replay.csv"), index=False)

    if not df_sum.empty:
        best_name = str(df_sum.loc[0, "rule_name"])
        df_best = df_all[df_all["rule_name"] == best_name].copy()
        df_best.to_csv(os.path.join(out_dir, f"season{season_num}_task4_newrule_replay.csv"), index=False)
        plot_task4_summary(df_sum, out_dir, season_num, best_name)
    else:
        df_best = pd.DataFrame()
        plot_task4_summary(df_sum, out_dir, season_num, best_rule_name="")

    return df_sum, df_best


# -----------------------------
# Main runner
# -----------------------------
def main():
    base_out = os.path.join(os.getcwd(), "outputs_task234")
    ensure_dir(base_out)

    data_path = "2026_MCM_Problem_C_Data.csv"
    random_seed = 123

    estimator = make_estimator(data_path=data_path, random_seed=random_seed)

    seasons = [1, 2, 3, 4, 5]

    for season_num in seasons:
        season_out = os.path.join(base_out, f"season{season_num}")
        ensure_dir(season_out)

        try:
            votes = estimator.estimate_votes(
                season_num=season_num,
                draws=1200,
                tune=1200,
                chains=4,
                cores=1,
                target_accept=0.97,
                random_seed=random_seed,
            )
        except Exception as e:
            print(f"[Skip] Season {season_num} failed: {repr(e)}")
            continue

        # Task2
        task2_compare_methods(estimator, season_num, votes, season_out)
        task2_fliprate_under_noise(estimator, season_num, votes, season_out)

        # Task3
        task3_build_panel(estimator, season_num, votes, season_out)

        # Task4++ (grid search)
        task4_grid_search(estimator, season_num, votes, season_out)

        print(f"[Done] Season {season_num} task234 outputs: {season_out}")

    print(f"\nAll Task2/3/4 outputs saved under: {base_out}")


if __name__ == "__main__":
    main()
