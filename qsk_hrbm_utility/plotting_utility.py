import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###################################################################################################
# Load and combine result files
###################################################################################################
def load_and_aggregate_metrics(data_dir="."):
    """Finds all CSV files for energy and entanglement across sizes N,
    combines them, and returns full raw DataFrames.
    """
    energy_files = glob.glob(
        os.path.join(data_dir, "**", "*_energy_metrics.csv"), recursive=True
    )
    entangle_files = glob.glob(
        os.path.join(data_dir, "**", "*_entanglement_metrics.csv"),
        recursive=True,
    )

    if not energy_files or not entangle_files:
        raise FileNotFoundError(f"No metric CSV files found in '{data_dir}'")

    df_energy = pd.concat(
        [pd.read_csv(f) for f in energy_files], ignore_index=True
    )
    df_entangle = pd.concat(
        [pd.read_csv(f) for f in entangle_files], ignore_index=True
    )

    return df_energy, df_entangle
###################################################################################################
# Exact exact E, S2, SvN
###################################################################################################
def export_exact_gs_metrics(data_dir=".", save_path="exact_gs_metrics.csv"):
    """Extracts and saves exact GS energy, S_2, and S_vN across all system sizes (N)
    and QSK disorder seeds.
    """
    df_energy, df_entangle = load_and_aggregate_metrics(data_dir)

    # Extract Exact GS Energy
    exact_e = (
        df_energy[df_energy["model"] == "Exact GS"][["N", "seed_qsk", "energy"]]
        .drop_duplicates(subset=["N", "seed_qsk"])
        .rename(columns={"energy": "exact_energy"})
    )

    # Extract Exact GS Entropies
    exact_s = (
        df_entangle[df_entangle["model"] == "Exact GS"][
            ["N", "seed_qsk", "S_2", "S_vN"]
        ]
        .drop_duplicates(subset=["N", "seed_qsk"])
        .rename(columns={"S_2": "exact_S_2", "S_vN": "exact_S_vN"})
    )

    # Merge into a single reference DataFrame
    exact_df = pd.merge(exact_e, exact_s, on=["N", "seed_qsk"], how="outer")
    exact_df = exact_df.sort_values(by=["N", "seed_qsk"]).reset_index(drop=True)

    exact_df.to_csv(save_path, index=False)
    print(f"Exact GS metrics successfully saved to: {save_path}")
    return exact_df

###################################################################################################
# Scaling benchmark per seed
###################################################################################################
def plot_scaling_benchmark_per_seed(
    data_dir=".", save_prefix="scaling_benchmark_seed"
):
    """Generates 6-panel scaling plots and corresponding CSV metric files for each QSK seed."""
    df_energy, df_entangle = load_and_aggregate_metrics(data_dir)

    df_e_nqs = df_energy[df_energy["model"] != "Exact GS"].copy()

    # Exact GS reference for per-seed absolute error computation
    exact_s = (
        df_entangle[df_entangle["model"] == "Exact GS"][
            ["N", "seed_qsk", "S_vN", "S_2"]
        ]
        .drop_duplicates(subset=["N", "seed_qsk"])
        .rename(columns={"S_vN": "S_vN_exact", "S_2": "S_2_exact"})
    )

    df_s_nqs = df_entangle[df_entangle["model"] != "Exact GS"].copy()
    df_s_nqs = pd.merge(df_s_nqs, exact_s, on=["N", "seed_qsk"], how="left")
    df_s_nqs["svn_err"] = (df_s_nqs["S_vN"] - df_s_nqs["S_vN_exact"]).abs()
    df_s_nqs["s2_err"] = (df_s_nqs["S_2"] - df_s_nqs["S_2_exact"]).abs()

    sem_fn = lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0.0

    # Aggregations per (N, seed_qsk, model)
    e_summary = (
        df_e_nqs.groupby(["N", "seed_qsk", "model"])
        .agg(
            mean_rel_err=("rel_energy_error", "mean"),
            sem_rel_err=("rel_energy_error", sem_fn),
            mean_var=("energy_variance", "mean"),
            sem_var=("energy_variance", sem_fn),
        )
        .reset_index()
    )

    s_all_summary = (
        df_entangle.groupby(["N", "seed_qsk", "model"])
        .agg(
            mean_svn=("S_vN", "mean"),
            sem_svn=("S_vN", sem_fn),
            mean_s2=("S_2", "mean"),
            sem_s2=("S_2", sem_fn),
        )
        .reset_index()
    )

    s_err_summary = (
        df_s_nqs.groupby(["N", "seed_qsk", "model"])
        .agg(
            mean_svn_err=("svn_err", "mean"),
            sem_svn_err=("svn_err", sem_fn),
            mean_s2_err=("s2_err", "mean"),
            sem_s2_err=("s2_err", sem_fn),
        )
        .reset_index()
    )

    seeds = sorted(df_energy["seed_qsk"].dropna().unique())
    styles = {
    "Exact GS": {"color": "black", "marker": "x", "ls": "-", "zorder": 5},
    # High zorder keeps black on top
    "RBM (Euclidean)": {"color": "purple", "marker": "o", "ls": "--", "zorder": 3},
    "HRBM (Hyperbolic)": {"color": "green", "marker": "s", "ls": "--", "zorder": 4},
    # Change to dotted line or lower zorder
    }

    for seed in seeds:
        seed_int = int(seed)
        e_seed = e_summary[e_summary["seed_qsk"] == seed]
        s_all_seed = s_all_summary[s_all_summary["seed_qsk"] == seed]
        s_err_seed = s_err_summary[s_err_summary["seed_qsk"] == seed]

        # Consolidated DataFrame for the current seed
        df_seed_csv = pd.merge(
            e_seed, s_all_seed, on=["N", "seed_qsk", "model"], how="outer"
        )
        df_seed_csv = pd.merge(
            df_seed_csv, s_err_seed, on=["N", "seed_qsk", "model"], how="outer"
        ).sort_values(by=["N", "model"])

        csv_path = f"{save_prefix}_{seed_int}_metrics.csv"
        df_seed_csv.to_csv(csv_path, index=False)
        print(f"Saved seed {seed_int} CSV metrics to: {csv_path}")

        # Plot generation
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        # Row 1, Col 1: Relative Energy Error
        ax = axes[0, 0]
        for model in ["RBM (Euclidean)", "HRBM (Hyperbolic)"]:
            sub = e_seed[e_seed["model"] == model]
            ax.errorbar(
                sub["N"],
                sub["mean_rel_err"],
                yerr=sub["sem_rel_err"],
                label=model,
                capsize=4,
                linewidth=1.8,
                **styles[model],
            )
        ax.set_title(
            rf"Relative Energy Error $\epsilon$ vs $N$ (Seed {seed_int})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("System Size $N$", fontsize=10)
        ax.set_ylabel(r"Relative Error $\epsilon(N)$", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.legend()

        # Row 1, Col 2: Rényi-2 Entropy (S_2)
        ax = axes[0, 1]
        for model in ["Exact GS", "RBM (Euclidean)", "HRBM (Hyperbolic)"]:
            sub = s_all_seed[s_all_seed["model"] == model]
            ax.errorbar(
                sub["N"],
                sub["mean_s2"],
                yerr=sub["sem_s2"],
                label=model,
                capsize=4,
                linewidth=1.8,
                **styles[model],
            )
        ax.set_title(
            f"Rényi-2 Entropy $S_2$ vs $N$ (Seed {seed_int})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("System Size $N$", fontsize=10)
        ax.set_ylabel(r"$S_2$", fontsize=10)
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend()

        # Row 1, Col 3: von Neumann Entropy (S_vN)
        ax = axes[0, 2]
        for model in ["Exact GS", "RBM (Euclidean)", "HRBM (Hyperbolic)"]:
            sub = s_all_seed[s_all_seed["model"] == model]
            ax.errorbar(
                sub["N"],
                sub["mean_svn"],
                yerr=sub["sem_svn"],
                label=model,
                capsize=4,
                linewidth=1.8,
                **styles[model],
            )
        ax.set_title(
            f"von Neumann Entropy $S_{{vN}}$ vs $N$ (Seed {seed_int})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("System Size $N$", fontsize=10)
        ax.set_ylabel(r"$S_{vN}$", fontsize=10)
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend()

        # Row 2, Col 1: Energy Variance
        ax = axes[1, 0]
        for model in ["RBM (Euclidean)", "HRBM (Hyperbolic)"]:
            sub = e_seed[e_seed["model"] == model]
            ax.errorbar(
                sub["N"],
                sub["mean_var"],
                yerr=sub["sem_var"],
                label=model,
                capsize=4,
                linewidth=1.8,
                **styles[model],
            )
        ax.set_title(
            rf"Energy Variance $\sigma^2$ vs $N$ (Seed {seed_int})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("System Size $N$", fontsize=10)
        ax.set_ylabel(r"Energy Variance $\sigma_E^2(N)$", fontsize=10)
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend()

        # Row 2, Col 2: Absolute Error in S_2
        ax = axes[1, 1]
        for model in ["RBM (Euclidean)", "HRBM (Hyperbolic)"]:
            sub = s_err_seed[s_err_seed["model"] == model]
            ax.errorbar(
                sub["N"],
                sub["mean_s2_err"],
                yerr=sub["sem_s2_err"],
                label=model,
                capsize=4,
                linewidth=1.8,
                **styles[model],
            )
        ax.set_title(
            f"Absolute Error in $S_2$ vs $N$ (Seed {seed_int})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("System Size $N$", fontsize=10)
        ax.set_ylabel(r"$|S_2^{NQS} - S_2^{Exact}|$", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.legend()

        # Row 2, Col 3: Absolute Error in S_vN
        ax = axes[1, 2]
        for model in ["RBM (Euclidean)", "HRBM (Hyperbolic)"]:
            sub = s_err_seed[s_err_seed["model"] == model]
            ax.errorbar(
                sub["N"],
                sub["mean_svn_err"],
                yerr=sub["sem_svn_err"],
                label=model,
                capsize=4,
                linewidth=1.8,
                **styles[model],
            )
        ax.set_title(
            f"Absolute Error in $S_{{vN}}$ vs $N$ (Seed {seed_int})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("System Size $N$", fontsize=10)
        ax.set_ylabel(r"$|S_{vN}^{NQS} - S_{vN}^{Exact}|$", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.legend()

        plt.tight_layout()
        save_path = f"{save_prefix}_{seed_int}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved seed scaling plot to: {save_path}")

###################################################################################################
# Scaling benchmark all seeds
###################################################################################################
def plot_scaling_benchmark_all_seeds(
    data_dir=".",
    save_path="scaling_benchmark.png",
    csv_save_path="scaling_benchmark_all_seeds_metrics.csv",
):
    """Generates an aggregated 6-panel scaling plot averaged across all QSK seeds
    and exports all plotted metrics to CSV.
    """
    df_energy, df_entangle = load_and_aggregate_metrics(data_dir)

    df_e_nqs = df_energy[df_energy["model"] != "Exact GS"].copy()

    exact_s = (
        df_entangle[df_entangle["model"] == "Exact GS"][
            ["N", "seed_qsk", "S_vN", "S_2"]
        ]
        .drop_duplicates(subset=["N", "seed_qsk"])
        .rename(columns={"S_vN": "S_vN_exact", "S_2": "S_2_exact"})
    )

    df_s_nqs = df_entangle[df_entangle["model"] != "Exact GS"].copy()
    df_s_nqs = pd.merge(df_s_nqs, exact_s, on=["N", "seed_qsk"], how="left")
    df_s_nqs["svn_err"] = (df_s_nqs["S_vN"] - df_s_nqs["S_vN_exact"]).abs()
    df_s_nqs["s2_err"] = (df_s_nqs["S_2"] - df_s_nqs["S_2_exact"]).abs()

    sem_fn = lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0.0

    # 1. Average energy metrics over NQS seeds within each QSK instance
    df_e_qsk_means = (
        df_e_nqs.groupby(["N", "seed_qsk", "model"])
        .agg(
            rel_energy_error=("rel_energy_error", "mean"),
            energy_variance=("energy_variance", "mean"),
        )
        .reset_index()
    )

    e_summary = (
        df_e_qsk_means.groupby(["N", "model"])
        .agg(
            mean_rel_err=("rel_energy_error", "mean"),
            sem_rel_err=("rel_energy_error", sem_fn),
            mean_var=("energy_variance", "mean"),
            sem_var=("energy_variance", sem_fn),
        )
        .reset_index()
    )

    # 2. Average entanglement metrics over NQS seeds within each QSK instance
    df_s_qsk_means = (
        df_entangle.groupby(["N", "seed_qsk", "model"])
        .agg(
            S_vN=("S_vN", "mean"),
            S_2=("S_2", "mean"),
        )
        .reset_index()
    )

    s_all_summary = (
        df_s_qsk_means.groupby(["N", "model"])
        .agg(
            mean_svn=("S_vN", "mean"),
            sem_svn=("S_vN", sem_fn),
            mean_s2=("S_2", "mean"),
            sem_s2=("S_2", sem_fn),
        )
        .reset_index()
    )

    # 3. Average entanglement errors over NQS seeds within each QSK instance
    df_s_err_qsk_means = (
        df_s_nqs.groupby(["N", "seed_qsk", "model"])
        .agg(
            svn_err=("svn_err", "mean"),
            s2_err=("s2_err", "mean"),
        )
        .reset_index()
    )

    s_err_summary = (
        df_s_err_qsk_means.groupby(["N", "model"])
        .agg(
            mean_svn_err=("svn_err", "mean"),
            sem_svn_err=("svn_err", sem_fn),
            mean_s2_err=("s2_err", "mean"),
            sem_s2_err=("s2_err", sem_fn),
        )
        .reset_index()
    )

    # Export aggregated plotting metrics to CSV
    df_all_summary = pd.merge(
        e_summary, s_all_summary, on=["N", "model"], how="outer"
    )
    df_all_summary = pd.merge(
        df_all_summary, s_err_summary, on=["N", "model"], how="outer"
    ).sort_values(by=["N", "model"])

    df_all_summary.to_csv(csv_save_path, index=False)
    print(f"Aggregated all-seeds CSV metrics saved to: {csv_save_path}")

    # Plot generation
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    styles = {
        "Exact GS": {"color": "black", "marker": "x", "ls": "-"},
        "RBM (Euclidean)": {"color": "purple", "marker": "o", "ls": "--"},
        "HRBM (Hyperbolic)": {"color": "green", "marker": "s", "ls": "-"},
    }

    # Row 1, Col 1: Relative Energy Error
    ax = axes[0, 0]
    for model in ["RBM (Euclidean)", "HRBM (Hyperbolic)"]:
        sub = e_summary[e_summary["model"] == model]
        ax.errorbar(
            sub["N"],
            sub["mean_rel_err"],
            yerr=sub["sem_rel_err"],
            label=model,
            capsize=4,
            linewidth=1.8,
            **styles[model],
        )
    ax.set_title(r"Relative Energy Error $\epsilon$ vs $N$", fontsize=11, fontweight="bold")
    ax.set_xlabel("System Size $N$", fontsize=10)
    ax.set_ylabel(r"Relative Error $\epsilon(N)$", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend()

    # Row 1, Col 2: Rényi-2 Entropy (S_2)
    ax = axes[0, 1]
    for model in ["Exact GS", "RBM (Euclidean)", "HRBM (Hyperbolic)"]:
        sub = s_all_summary[s_all_summary["model"] == model]
        ax.errorbar(
            sub["N"],
            sub["mean_s2"],
            yerr=sub["sem_s2"],
            label=model,
            capsize=4,
            linewidth=1.8,
            **styles[model],
        )
    ax.set_title(
        r"Rényi-2 Entropy $S_2$ vs $N$", fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("System Size $N$", fontsize=10)
    ax.set_ylabel(r"$S_2$", fontsize=10)
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend()

    # Row 1, Col 3: von Neumann Entropy (S_vN)
    ax = axes[0, 2]
    for model in ["Exact GS", "RBM (Euclidean)", "HRBM (Hyperbolic)"]:
        sub = s_all_summary[s_all_summary["model"] == model]
        ax.errorbar(
            sub["N"],
            sub["mean_svn"],
            yerr=sub["sem_svn"],
            label=model,
            capsize=4,
            linewidth=1.8,
            **styles[model],
        )
    ax.set_title(
        r"von Neumann Entropy $S_{vN}$ vs $N$", fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("System Size $N$", fontsize=10)
    ax.set_ylabel(r"$S_{vN}$", fontsize=10)
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend()

    # Row 2, Col 1: Energy Variance
    ax = axes[1, 0]
    for model in ["RBM (Euclidean)", "HRBM (Hyperbolic)"]:
        sub = e_summary[e_summary["model"] == model]
        ax.errorbar(
            sub["N"],
            sub["mean_var"],
            yerr=sub["sem_var"],
            label=model,
            capsize=4,
            linewidth=1.8,
            **styles[model],
        )
    ax.set_title(rf"Energy Variance $\sigma^2$ vs $N$", fontsize=11, fontweight="bold")
    ax.set_xlabel("System Size $N$", fontsize=10)
    ax.set_ylabel(r"Energy Variance $\sigma_E^2(N)$", fontsize=10)
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend()

    # Row 2, Col 2: Absolute Error in S_2
    ax = axes[1, 1]
    for model in ["RBM (Euclidean)", "HRBM (Hyperbolic)"]:
        sub = s_err_summary[s_err_summary["model"] == model]
        ax.errorbar(
            sub["N"],
            sub["mean_s2_err"],
            yerr=sub["sem_s2_err"],
            label=model,
            capsize=4,
            linewidth=1.8,
            **styles[model],
        )
    ax.set_title(
        r"Absolute Error in $S_2$ vs $N$", fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("System Size $N$", fontsize=10)
    ax.set_ylabel(r"$|S_2^{NQS} - S_2^{Exact}|$", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend()

    # Row 2, Col 3: Absolute Error in S_vN
    ax = axes[1, 2]
    for model in ["RBM (Euclidean)", "HRBM (Hyperbolic)"]:
        sub = s_err_summary[s_err_summary["model"] == model]
        ax.errorbar(
            sub["N"],
            sub["mean_svn_err"],
            yerr=sub["sem_svn_err"],
            label=model,
            capsize=4,
            linewidth=1.8,
            **styles[model],
        )
    ax.set_title(
        r"Absolute Error in $S_{vN}$ vs $N$", fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("System Size $N$", fontsize=10)
    ax.set_ylabel(r"$|S_{vN}^{NQS} - S_{vN}^{Exact}|$", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Scaling figure successfully saved to: {save_path}")


if __name__ == "__main__":
    # Generate exact ground truth CSV reference
    export_exact_gs_metrics(save_path="exact_gs_metrics.csv")

    # Save per-seed CSV metrics and plots
    plot_scaling_benchmark_per_seed(save_prefix="scaling_benchmark_seed")

    # Save aggregated all-seeds CSV metrics and benchmark plot
    plot_scaling_benchmark_all_seeds(
        save_path="scaling_benchmark.png",
        csv_save_path="scaling_benchmark_all_seeds_metrics.csv",
    )
