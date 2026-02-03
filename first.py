# first.py
# -*- coding: utf-8 -*-

import os
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class LowDimBayesVoteEstimator:
    """
    Low-dimensional, stable Bayesian model using elimination outcomes only.

    Key idea:
    - Each contestant i has a season-level latent popularity baseline mu_i.
    - Weekly vote share is the softmax over mu_i among contestants still "alive" that week:
        V[t,:] = softmax(mu_i + mask_penalty[t,:])
      where mask_penalty[t,i] is 0 if alive, large negative if not alive.
    - Combined score:
        S[t,i] = alpha_w * J_norm[t,i] + (1 - alpha_w) * V[t,i]
    - Elimination likelihood (for each elimination event k at week index t_k):
        elim_obs[k] ~ Categorical( softmax( -tau * S[t_k,:] + mask_penalty[t_k,:] ) )

    This is intentionally low-dimensional to avoid identifiability issues from high-dimensional eta[t,i]
    under sparse elimination-only observations.
    """

    def __init__(self, data_path="2026_MCM_Problem_C_Data.csv", random_seed=42):
        self.data = pd.read_csv(data_path)
        self.random_seed = int(random_seed)

        self.models = {}
        self.traces = {}
        self.vote_estimates = {}
        self.processed_cache = {}

    # ----------------------------
    # 1) Preprocess
    # ----------------------------
    def preprocess_data(self, season_num: int) -> dict:
        season_num = int(season_num)
        if season_num in self.processed_cache:
            return self.processed_cache[season_num]

        season_df = self.data[self.data["season"] == season_num].copy()
        if season_df.empty:
            raise ValueError(f"No rows found for season={season_num}")

        celebrities = season_df["celebrity_name"].astype(str).unique()
        n = len(celebrities)

        week_cols = [
            c for c in season_df.columns
            if c.startswith("week") and ("judge" in c) and c.endswith("_score")
        ]
        if not week_cols:
            raise ValueError("No week*_judge*_score columns found in CSV.")

        week_numbers = sorted({int(c.split("_")[0].replace("week", "")) for c in week_cols})
        T = len(week_numbers)

        # row lookup per celebrity (assumes one row per celeb per season)
        celeb_row = {}
        for celeb in celebrities:
            rows = season_df[season_df["celebrity_name"].astype(str) == celeb]
            if rows.empty:
                raise ValueError(f"Missing row for celeb={celeb} in season={season_num}")
            celeb_row[celeb] = rows.iloc[0]

        # Raw judges totals: J[t,i]
        J = np.zeros((T, n), dtype=float)
        for i, celeb in enumerate(celebrities):
            row = celeb_row[celeb]
            for t_idx, w in enumerate(week_numbers):
                s = 0.0
                seen = False
                for judge_num in (1, 2, 3, 4):
                    col = f"week{w}_judge{judge_num}_score"
                    if col in season_df.columns:
                        v = row[col]
                        if not pd.isna(v):
                            s += float(v)
                            seen = True
                J[t_idx, i] = s if seen else 0.0

        # last active week index per celeb: last t where J[t,i] > 0
        last_pos = np.full(n, -1, dtype=int)
        for i in range(n):
            pos = np.where(J[:, i] > 0)[0]
            last_pos[i] = int(pos.max()) if len(pos) else -1

        season_max_week_pos = int(np.max(last_pos)) if np.max(last_pos) >= 0 else -1

        # alive mask: based on last_pos, and (optionally) J>0
        alive_mask = np.zeros_like(J, dtype=bool)
        for i in range(n):
            if last_pos[i] >= 0:
                alive_mask[: last_pos[i] + 1, i] = True

        # Conservative: require J>0 to be considered active
        alive_mask &= (J > 0)

        # Elimination events: eliminated at last_pos if not finalist
        elim_weeks_idx = []
        elim_obs = []
        for i in range(n):
            if last_pos[i] >= 0 and last_pos[i] < season_max_week_pos:
                elim_weeks_idx.append(last_pos[i])
                elim_obs.append(i)

        if elim_weeks_idx:
            order = np.argsort(elim_weeks_idx)
            elim_weeks_idx = np.array(elim_weeks_idx, dtype=int)[order]
            elim_obs = np.array(elim_obs, dtype=int)[order]
        else:
            elim_weeks_idx = np.array([], dtype=int)
            elim_obs = np.array([], dtype=int)

        # Min-max normalize judge scores per week among alive contestants
        J_norm = np.zeros_like(J, dtype=float)
        for t in range(T):
            idx = np.where(alive_mask[t])[0]
            if len(idx) <= 1:
                continue
            vals = J[t, idx]
            mn, mx = float(vals.min()), float(vals.max())
            if mx > mn:
                J_norm[t, idx] = (vals - mn) / (mx - mn)
            else:
                J_norm[t, idx] = 0.0

        processed = {
            "season": season_num,
            "celebrities": celebrities,
            "weeks": week_numbers,
            "J": J,
            "J_norm": J_norm,
            "alive_mask": alive_mask,
            "last_pos": last_pos,
            "season_max_week_pos": season_max_week_pos,
            "elim_weeks_idx": elim_weeks_idx,
            "elim_obs": elim_obs,
        }
        self.processed_cache[season_num] = processed
        return processed

    # ----------------------------
    # 2) Build low-dimensional model
    # ----------------------------
    def build_model(
        self,
        season_data: dict,
        tau_prior: float = 5.0,
        alpha_prior_a: float = 2.0,
        alpha_prior_b: float = 2.0,
        sigma_mu_prior: float = 0.8,
        mask_strength: float = 30.0,
    ) -> pm.Model:
        J_norm = season_data["J_norm"]         # (T,n)
        alive_mask = season_data["alive_mask"] # (T,n)
        elim_t = season_data["elim_weeks_idx"] # (K,)
        elim_i = season_data["elim_obs"]       # (K,)

        T, n = J_norm.shape

        mask_penalty = np.where(alive_mask, 0.0, -float(mask_strength)).astype(float)

        with pm.Model() as model:
            mu0 = pm.Normal("mu0", mu=0.0, sigma=1.5)
            sigma_mu = pm.HalfNormal("sigma_mu", sigma=sigma_mu_prior)
            z_mu = pm.Normal("z_mu", mu=0.0, sigma=1.0, shape=n)
            mu_i = pm.Deterministic("mu_i", mu0 + sigma_mu * z_mu)  # (n,)

            V = pm.Deterministic("V", pm.math.softmax(mu_i + mask_penalty, axis=1))

            alpha_w = pm.Beta("alpha_w", alpha=alpha_prior_a, beta=alpha_prior_b)
            tau = pm.HalfNormal("tau", sigma=tau_prior)

            Jn = pm.Data("J_norm", J_norm)
            S = pm.Deterministic("S", alpha_w * Jn + (1.0 - alpha_w) * V)

            if elim_t.size > 0:
                logits = -tau * S[elim_t, :] + mask_penalty[elim_t, :]
                p_elim = pm.Deterministic("p_elim", pm.math.softmax(logits, axis=1))
                pm.Categorical("elim_obs", p=p_elim, observed=elim_i)
            else:
                pm.Potential("no_elim_potential", 0.0)

        return model

    # ----------------------------
    # 3) Fit model & extract weekly vote shares
    # ----------------------------
    def estimate_votes(
        self,
        season_num: int,
        draws: int = 1200,
        tune: int = 1200,
        chains: int = 4,
        cores: int = 1,
        target_accept: float = 0.97,
        max_treedepth: int = 12,
        init: str = "adapt_diag",
        random_seed: int | None = None,
        model_kwargs: dict | None = None,
    ) -> dict:
        season_num = int(season_num)
        seed = self.random_seed if random_seed is None else int(random_seed)
        model_kwargs = model_kwargs or {}

        print(f"[Bayes] Estimating vote shares for season {season_num}...")

        season_data = self.preprocess_data(season_num)
        model = self.build_model(season_data, **model_kwargs)
        self.models[season_num] = model

        with model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                max_treedepth=max_treedepth,
                init=init,
                random_seed=seed,
                return_inferencedata=True,
                progressbar=True,
            )

        self.traces[season_num] = trace

        V_post = trace.posterior["V"].values
        V_flat = V_post.reshape(-1, V_post.shape[-2], V_post.shape[-1])

        vote_estimates = {}
        weeks = season_data["weeks"]
        celebrities = season_data["celebrities"]

        for t_idx, week in enumerate(weeks):
            vote_estimates[week] = {}
            for i, celeb in enumerate(celebrities):
                samples = V_flat[:, t_idx, i]
                vote_estimates[week][celeb] = {
                    "mean": float(np.mean(samples)),
                    "median": float(np.median(samples)),
                    "std": float(np.std(samples)),
                    "q05": float(np.percentile(samples, 5)),
                    "q95": float(np.percentile(samples, 95)),
                }

        self.vote_estimates[season_num] = vote_estimates
        print(f"[Bayes] Season {season_num} finished. Stored posterior vote-share summaries.")
        return vote_estimates

    # ----------------------------
    # 4) Simple validation
    # ----------------------------
    def validate_model(self, season_num: int) -> dict | None:
        season_num = int(season_num)
        if season_num not in self.vote_estimates:
            print(f"Please run estimate_votes({season_num}) first.")
            return None

        season_data = self.preprocess_data(season_num)
        vote_est = self.vote_estimates[season_num]

        elim_t = season_data["elim_weeks_idx"]
        elim_i = season_data["elim_obs"]
        weeks = season_data["weeks"]
        celebs = season_data["celebrities"]
        J_norm = season_data["J_norm"]
        alive = season_data["alive_mask"]

        alpha_hat = float(self.traces[season_num].posterior["alpha_w"].values.mean()) if season_num in self.traces else 0.5

        results = {"correct_predictions": 0, "total_eliminations": 0, "predictions": []}

        for k in range(len(elim_t)):
            t = int(elim_t[k])
            actual = int(elim_i[k])
            week_label = weeks[t]

            V_mean = np.array([vote_est[week_label][c]["mean"] for c in celebs], dtype=float)
            eligible = alive[t].copy()
            if eligible.sum() <= 1:
                continue

            S_hat = alpha_hat * J_norm[t] + (1.0 - alpha_hat) * V_mean
            S_hat = np.where(eligible, S_hat, np.inf)

            pred = int(np.argmin(S_hat))
            ok = (pred == actual)

            results["predictions"].append({
                "week": int(week_label),
                "actual": str(celebs[actual]),
                "predicted": str(celebs[pred]),
                "correct": bool(ok),
            })
            results["total_eliminations"] += 1
            if ok:
                results["correct_predictions"] += 1

        results["accuracy"] = (
            results["correct_predictions"] / results["total_eliminations"]
            if results["total_eliminations"] > 0 else 0.0
        )
        return results


# ----------------------------
# Output helpers
# ----------------------------
def save_vote_share_csv(votes: dict, season_num: int, out_dir: str):
    rows = []
    for week, wk_data in votes.items():
        for celeb, d in wk_data.items():
            rows.append({
                "season": season_num,
                "week": week,
                "celebrity": celeb,
                "vote_share_mean": d["mean"],
                "vote_share_median": d["median"],
                "vote_share_std": d["std"],
                "vote_share_q05": d["q05"],
                "vote_share_q95": d["q95"],
            })
    out = pd.DataFrame(rows).sort_values(["week", "vote_share_mean"], ascending=[True, False])
    path = os.path.join(out_dir, f"season{season_num}_vote_share_estimates.csv")
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {path}")


def save_param_summary_csv(trace, season_num: int, out_dir: str):
    summ = az.summary(trace, var_names=["alpha_w", "tau", "sigma_mu"], round_to=4)
    path = os.path.join(out_dir, f"season{season_num}_param_summary.csv")
    summ.to_csv(path, encoding="utf-8-sig")
    print(f"[Saved] {path}")


def save_diagnostics_plot(trace, season_num: int, out_dir: str):
    az.plot_trace(trace, var_names=["alpha_w", "tau", "sigma_mu"])
    plt.tight_layout()
    path = os.path.join(out_dir, f"season{season_num}_diagnostics.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[Saved] {path}")


def save_week_vote_plot(estimator: LowDimBayesVoteEstimator, season_num: int, week: int, out_dir: str, top_k: int = 15):
    season_data = estimator.preprocess_data(season_num)
    votes = estimator.vote_estimates[season_num][week]

    celebs_sorted = sorted(votes.keys(), key=lambda c: votes[c]["mean"], reverse=True)[:top_k]
    means = np.array([votes[c]["mean"] for c in celebs_sorted], dtype=float)
    q05 = np.array([votes[c]["q05"] for c in celebs_sorted], dtype=float)
    q95 = np.array([votes[c]["q95"] for c in celebs_sorted], dtype=float)

    x = np.arange(len(celebs_sorted))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    yerr = np.vstack([means - q05, q95 - means])
    axes[0].bar(x, means, yerr=yerr, capsize=5, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(celebs_sorted, rotation=45, ha="right")
    axes[0].set_ylabel("Estimated Vote Share (posterior mean)")
    axes[0].set_title(f"Season {season_num}, Week {week}: Vote Share with 90% CI")
    axes[0].grid(True, alpha=0.3)

    weeks = season_data["weeks"]
    t_idx = weeks.index(week)
    J_raw = season_data["J"][t_idx]
    V_mean_all = np.array(
        [estimator.vote_estimates[season_num][week][c]["mean"] for c in season_data["celebrities"]],
        dtype=float
    )
    axes[1].scatter(J_raw, V_mean_all, alpha=0.8)
    axes[1].set_xlabel("Judge Total Score (raw)")
    axes[1].set_ylabel("Estimated Vote Share (posterior mean)")
    axes[1].set_title("Judge Scores vs Vote Share (posterior mean)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"season{season_num}_week{week}_votes.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[Saved] {path}")


def main():
    # 统一总输出目录：outputs/season1 ... outputs/season5
    base_out = os.path.join(os.getcwd(), "outputs")
    os.makedirs(base_out, exist_ok=True)

    estimator = LowDimBayesVoteEstimator(
        data_path="2026_MCM_Problem_C_Data.csv",
        random_seed=42,
    )

    seasons = [1, 2, 3, 4, 5]

    for season_num in seasons:
        season_out = os.path.join(base_out, f"season{season_num}")
        os.makedirs(season_out, exist_ok=True)

        try:
            votes = estimator.estimate_votes(
                season_num=season_num,
                draws=1200,
                tune=1200,
                chains=4,
                cores=1,
                target_accept=0.97,
                max_treedepth=12,
                init="adapt_diag",
                random_seed=42,
                model_kwargs=dict(
                    tau_prior=5.0,
                    alpha_prior_a=2.0,
                    alpha_prior_b=2.0,
                    sigma_mu_prior=0.8,
                    mask_strength=30.0,
                ),
            )
        except Exception as e:
            print(f"[Skip] Season {season_num} failed: {repr(e)}")
            continue

        trace = estimator.traces[season_num]

        # Save deliverables
        save_vote_share_csv(votes, season_num, season_out)
        save_param_summary_csv(trace, season_num, season_out)
        save_diagnostics_plot(trace, season_num, season_out)

        season_data = estimator.preprocess_data(season_num)
        if len(season_data["weeks"]) > 0:
            week_to_plot = season_data["weeks"][0]
            save_week_vote_plot(estimator, season_num, week_to_plot, season_out, top_k=15)

        # Validation
        val = estimator.validate_model(season_num)
        if val:
            print(f"Season {season_num} validation accuracy: {val['accuracy']:.2%} "
                  f"({val['correct_predictions']}/{val['total_eliminations']})")

        # Key diagnostics
        summ = az.summary(trace, var_names=["alpha_w", "tau", "sigma_mu"], round_to=4)
        print(f"\nSeason {season_num} key parameter diagnostics:")
        print(summ[["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]])

        print(f"[Done] Season {season_num} outputs: {season_out}\n")

    print(f"\nAll seasons finished. Outputs under: {base_out}")


if __name__ == "__main__":
    main()
