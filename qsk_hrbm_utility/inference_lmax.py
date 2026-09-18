import glob
import os
import re
import flax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
import pandas as pd
from hrbm import HyperbolicRBM
from qsk_netket_pipeline import *

# Try importing sk_cpu first to avoid JAX GPU OOM errors on N>=24
try:
    from sk_cpu import (
        generate_hamiltonian_sk,
        generate_hamiltonian_sk_scipy,
        generate_state_sk,
    )
except ImportError:
    from sk import generate_hamiltonian_sk, generate_state_sk

    generate_hamiltonian_sk_scipy = None


def find_checkpoint_dir(base_dir, N, seed_qsk, seed_nqs):
    """Dynamically searches Kaggle/local directory patterns to locate checkpoint directories."""
    if base_dir == "." and os.path.exists("/kaggle/working/qsk_hrbm"):
        base_dir = "/kaggle/working/qsk_hrbm"

    candidate_paths = [
        os.path.join(
            base_dir,
            f"N={N}",
            f"qsk_N={N}",
            f"seed_qsk={seed_qsk}",
            f"seed={seed_nqs}",
        ),
        os.path.join(
            base_dir, f"qsk_N={N}", f"seed_qsk={seed_qsk}", f"seed={seed_nqs}"
        ),
        os.path.join(
            base_dir, f"N={N}", f"seed_qsk={seed_qsk}", f"seed={seed_nqs}"
        ),
        os.path.join(
            base_dir, f"qsk_N={N}", f"seed_qsk={seed_qsk}_seed_nqs={seed_nqs}"
        ),
        os.path.join(
            base_dir, f"qsk_N={N}", f"qsk_seed={seed_qsk}_nqs_seed={seed_nqs}"
        ),
        os.path.join(base_dir, f"qsk_seed={seed_qsk}_nqs_seed={seed_nqs}"),
        os.path.join(base_dir, f"qsk_seed={seed_qsk}", f"nqs_seed={seed_nqs}"),
        os.path.join(base_dir, f"qsk_N={N}"),
        base_dir,
    ]

    for p in candidate_paths:
        if os.path.exists(p) and any(
            f.endswith(".mpack") for f in os.listdir(p)
        ):
            return p

    raise FileNotFoundError(
        f"Could not locate .mpack checkpoint files for N={N}, seed_qsk={seed_qsk}, seed_nqs={seed_nqs} "
        f"under base directory '{base_dir}'."
    )


def discover_lmax_values(chkpt_path, alpha=1):
    """Scans the checkpoint path for available hrbm_Lmax=* parameter files."""
    lmax_set = set()
    if os.path.exists(chkpt_path):
        for fname in os.listdir(chkpt_path):
            match = re.search(
                rf"hrbm_Lmax=(\d+)_alpha={alpha}_ML_final_params\.mpack", fname
            )
            if not match:
                match = re.search(
                    rf"hrbm_alpha={alpha}_Lmax=(\d+)_ML_final_params\.mpack",
                    fname,
                )
            if match:
                lmax_set.add(int(match.group(1)))
    return sorted(list(lmax_set))


def calc_exact_ee(N, seed_qsk, alpha):
    hi, H = generate_hamiltonian_sk(N=N, seed=seed_qsk)

    if generate_hamiltonian_sk_scipy is not None:
        sp_h = generate_hamiltonian_sk_scipy(N, seed_qsk)
    else:
        sp_h = H.to_sparse()

    _, ground_state_vec = generate_state_sk(N=N, seed=seed_qsk)
    exact_E = (ground_state_vec.conj().T @ sp_h @ ground_state_vec).real
    print(f"Exact ED Ground State Energy: {exact_E:.6f}")

    exact_s2, exact_svn, exact_spec = calculate_entanglement(
        ground_state_vec, N
    )
    return hi, exact_s2, exact_svn, exact_spec


def load_rbm_vstate(N, hi, chkpt_path, alpha):
    model_rbm = nk.models.RBM(alpha=alpha)
    vstate_rbm = nk.vqs.MCState(
        sampler=nk.sampler.MetropolisLocal(hi), model=model_rbm, n_samples=1008
    )

    rbm_file = os.path.join(
        chkpt_path, f"rbm_alpha={alpha}_ML_final_params.mpack"
    )
    if not os.path.exists(rbm_file):
        rbm_file = os.path.join(chkpt_path, "rbm_alpha=1_ML_final_params.mpack")

    with open(rbm_file, "rb") as f:
        vstate_rbm.parameters = flax.serialization.from_bytes(
            vstate_rbm.parameters, f.read()
        )
    return vstate_rbm


def load_hrbm_vstate(N, hi, chkpt_path, alpha, Lmax):
    model_hrbm = HyperbolicRBM(alpha=alpha, k=1.0, Lmax=Lmax)
    vstate_hrbm = nk.vqs.MCState(
        sampler=nk.sampler.MetropolisLocal(hi),
        model=model_hrbm,
        n_samples=1008,
    )

    hrbm_file = os.path.join(
        chkpt_path, f"hrbm_Lmax={Lmax}_alpha={alpha}_ML_final_params.mpack"
    )
    if not os.path.exists(hrbm_file):
        hrbm_file = os.path.join(
            chkpt_path, f"hrbm_alpha={alpha}_Lmax={Lmax}_ML_final_params.mpack"
        )

    with open(hrbm_file, "rb") as f:
        vstate_hrbm.parameters = flax.serialization.from_bytes(
            vstate_hrbm.parameters, f.read()
        )
    return vstate_hrbm


def compute_exact_vstate_metrics(vstate, H_sparse, exact_E):
    psi = np.array(vstate.to_array())
    psi = psi / np.linalg.norm(psi)

    H_psi = H_sparse @ psi
    eval_E = float(np.real(psi.conj().T @ H_psi))

    H2_psi = H_sparse @ H_psi
    eval_E2 = float(np.real(psi.conj().T @ H2_psi))

    variance = max(0.0, eval_E2 - eval_E**2)
    rel_error = float(np.abs(eval_E - exact_E) / np.abs(exact_E))

    return eval_E, rel_error, variance


# ========================================================================================
# Energy properties
# ========================================================================================
def evaluate_and_export_energies(
    N,
    alpha,
    seed_qsk_list,
    seed_nqs_list,
    lmax_list=None,
    base_dir=".",
    save_dir=None,
):
    if save_dir is None:
        save_dir = f"qsk_N={N}"
    os.makedirs(save_dir, exist_ok=True)

    energy_rows = []

    for seed_qsk in seed_qsk_list:
        print(f"Current QSK = {seed_qsk}")
        hi, H = generate_hamiltonian_sk(N=N, seed=seed_qsk)

        if generate_hamiltonian_sk_scipy is not None:
            H_sparse = generate_hamiltonian_sk_scipy(N, seed_qsk)
        else:
            H_sparse = H.to_sparse()

        _, ground_state_vec = generate_state_sk(N=N, seed=seed_qsk)
        exact_E = float(
            (ground_state_vec.conj().T @ H_sparse @ ground_state_vec).real
        )

        energy_rows.append(
            {
                "N": N,
                "seed_qsk": seed_qsk,
                "seed_nqs": None,
                "model": "Exact GS",
                "energy": exact_E,
                "rel_energy_error": 0.0,
                "energy_variance": 0.0,
            }
        )

        for seed_nqs in seed_nqs_list:
            chkpt_path = find_checkpoint_dir(base_dir, N, seed_qsk, seed_nqs)

            # 1. RBM Evaluation
            rbm_vstate = load_rbm_vstate(N, hi, chkpt_path, alpha)
            r_E, r_err, r_var = compute_exact_vstate_metrics(
                rbm_vstate, H_sparse, exact_E
            )
            energy_rows.append(
                {
                    "N": N,
                    "seed_qsk": seed_qsk,
                    "seed_nqs": seed_nqs,
                    "model": "RBM",
                    "energy": r_E,
                    "rel_energy_error": r_err,
                    "energy_variance": r_var,
                }
            )

            # 2. HRBM Evaluation for all Lmax values
            active_lmax_list = (
                lmax_list
                if lmax_list is not None
                else discover_lmax_values(chkpt_path, alpha)
            )

            for Lmax in active_lmax_list:
                hrbm_vstate = load_hrbm_vstate(
                    N, hi, chkpt_path, alpha, Lmax=Lmax
                )
                h_E, h_err, h_var = compute_exact_vstate_metrics(
                    hrbm_vstate, H_sparse, exact_E
                )
                energy_rows.append(
                    {
                        "N": N,
                        "seed_qsk": seed_qsk,
                        "seed_nqs": seed_nqs,
                        "model": f"HRBM (Lmax={Lmax})",
                        "energy": h_E,
                        "rel_energy_error": h_err,
                        "energy_variance": h_var,
                    }
                )

    df_energy = pd.DataFrame(energy_rows)
    csv_path = f"{save_dir}/qsk_N={N}_energy_metrics.csv"
    df_energy.to_csv(csv_path, index=False)
    print(f"Energy metrics saved successfully to: {csv_path}")

    df_summary = (
        df_energy.groupby("model")
        .agg(
            energy_mean=("energy", "mean"),
            rel_error_mean=("rel_energy_error", "mean"),
            rel_error_std=("rel_energy_error", "std"),
            variance_mean=("energy_variance", "mean"),
            variance_std=("energy_variance", "std"),
        )
        .reset_index()
    )

    return df_energy, df_summary


# ========================================================================================
# Entanglement properties
# ========================================================================================
def entanglement_extraction(
    N,
    alpha,
    seed_qsk_list,
    seed_nqs_list,
    lmax_list=None,
    base_dir=".",
    save_dir=None,
):
    if save_dir is None:
        save_dir = f"qsk_N={N}"
    os.makedirs(save_dir, exist_ok=True)

    metrics_rows = []
    spectra_rows = []

    for seed_qsk in seed_qsk_list:
        hi, exact_s2, exact_svn, exact_spec = calc_exact_ee(N, seed_qsk, alpha)
        exact_sorted = np.sort(exact_spec)[::-1]

        metrics_rows.append(
            {
                "N": N,
                "seed_qsk": seed_qsk,
                "seed_nqs": None,
                "model": "Exact GS",
                "S_2": exact_s2,
                "S_vN": exact_svn,
            }
        )

        for seed_nqs in seed_nqs_list:
            chkpt_path = find_checkpoint_dir(base_dir, N, seed_qsk, seed_nqs)

            # 1. RBM Entanglement Evaluation
            rbm_vstate = load_rbm_vstate(N, hi, chkpt_path, alpha)
            r_s2, r_svn, r_spec = calculate_entanglement(rbm_vstate, N)
            r_sorted = np.sort(r_spec)[::-1]

            metrics_rows.append(
                {
                    "N": N,
                    "seed_qsk": seed_qsk,
                    "seed_nqs": seed_nqs,
                    "model": "RBM",
                    "S_2": r_s2,
                    "S_vN": r_svn,
                }
            )

            # 2. HRBM Entanglement Evaluation for all Lmax values
            active_lmax_list = (
                lmax_list
                if lmax_list is not None
                else discover_lmax_values(chkpt_path, alpha)
            )

            for Lmax in active_lmax_list:
                hrbm_vstate = load_hrbm_vstate(
                    N, hi, chkpt_path, alpha, Lmax=Lmax
                )
                h_s2, h_svn, h_spec = calculate_entanglement(hrbm_vstate, N)
                h_sorted = np.sort(h_spec)[::-1]

                metrics_rows.append(
                    {
                        "N": N,
                        "seed_qsk": seed_qsk,
                        "seed_nqs": seed_nqs,
                        "model": f"HRBM (Lmax={Lmax})",
                        "S_2": h_s2,
                        "S_vN": h_svn,
                    }
                )

                for idx in range(len(exact_sorted)):
                    spectra_rows.append(
                        {
                            "N": N,
                            "Lmax": Lmax,
                            "seed_qsk": seed_qsk,
                            "seed_nqs": seed_nqs,
                            "schmidt_index": idx + 1,
                            "exact_lambda": exact_sorted[idx],
                            "rbm_lambda": r_sorted[idx],
                            "hrbm_lambda": h_sorted[idx],
                        }
                    )

    df_metrics = pd.DataFrame(metrics_rows)
    df_spectra = pd.DataFrame(spectra_rows)

    metrics_csv_path = f"{save_dir}/qsk_N={N}_entanglement_metrics.csv"
    spectra_csv_path = f"{save_dir}/qsk_N={N}_entanglement_spectra.csv"

    df_metrics.to_csv(metrics_csv_path, index=False)
    df_spectra.to_csv(spectra_csv_path, index=False)

    df_summary = (
        df_metrics.groupby("model")
        .agg(
            S_2_mean=("S_2", "mean"),
            S_2_std=("S_2", "std"),
            S_vN_mean=("S_vN", "mean"),
            S_vN_std=("S_vN", "std"),
        )
        .reset_index()
    )

    return df_metrics, df_spectra, df_summary

# ========================================================================================
# Plot entanglement spectrum (No Lmax)
# ========================================================================================
def plot_spectrum_from_csv(N, data_dir=None, save_dir=None):
    """Loads extracted entanglement spectrum CSV data and generates a multi-seed

    comparison plot across QSK realizations for a given system size N.
    """
    if data_dir is None:
        data_dir = f"qsk_N={N}"
    if save_dir is None:
        save_dir = data_dir

    csv_path = f"{data_dir}/qsk_N={N}_entanglement_spectra.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV file at: {csv_path}")

    # 1. Load CSV data
    df = pd.read_csv(csv_path)

    qsk_seeds = df["seed_qsk"].unique()
    n_seeds = len(qsk_seeds)

    # 2. Setup 3x2 Subplots Grid
    nrows, ncols = 3, 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 3.8 * nrows), sharey=True
    )
    axes_flat = axes.flatten()

    handles = []

    # 3. Iterate over QSK disorder seeds
    for idx, seed_qsk in enumerate(qsk_seeds[: nrows * ncols]):
        ax = axes_flat[idx]
        df_seed = df[df["seed_qsk"] == seed_qsk]

        # Aggregate across NQS initialization seeds
        grouped = df_seed.groupby("schmidt_index").agg(
            exact_lambda=("exact_lambda", "first"),
            rbm_mean=("rbm_lambda", "mean"),
            rbm_std=("rbm_lambda", "std"),
            hrbm_mean=("hrbm_lambda", "mean"),
            hrbm_std=("hrbm_lambda", "std"),
        )

        indices = grouped.index.values

        # Plot Exact GS
        (line1,) = ax.semilogy(
            indices,
            grouped["exact_lambda"],
            "k-",
            label="Exact GS",
            linewidth=2.0,
        )

        # Plot RBM Mean + Shaded Error Band
        (line2,) = ax.semilogy(
            indices,
            grouped["rbm_mean"],
            "r--",
            label=r"RBM (Euclidean) $\pm 1\sigma$",
            alpha=0.85,
            linewidth=1.5,
        )
        ax.fill_between(
            indices,
            np.clip(grouped["rbm_mean"] - grouped["rbm_std"], 1e-15, None),
            grouped["rbm_mean"] + grouped["rbm_std"],
            color="red",
            alpha=0.15,
        )

        # Plot HRBM Mean + Shaded Error Band
        (line3,) = ax.semilogy(
            indices,
            grouped["hrbm_mean"],
            "b-.",
            label=r"HRBM (Hyperbolic) $\pm 1\sigma$",
            alpha=0.85,
            linewidth=1.5,
        )
        ax.fill_between(
            indices,
            np.clip(grouped["hrbm_mean"] - grouped["hrbm_std"], 1e-15, None),
            grouped["hrbm_mean"] + grouped["hrbm_std"],
            color="blue",
            alpha=0.15,
        )

        if idx == 0:
            handles = [line1, line2, line3]

        # Apply y-axis label to left-column subplots
        if idx % ncols == 0:
            ax.set_ylabel(r"Eigenvalue $\lambda_i$", fontsize=11)

        ax.set_title(f"QSK Realization (Seed {seed_qsk})", fontsize=11)
        ax.set_xlabel("Schmidt Rank Index $i$", fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.35)

    # Clean up empty subplots if CSV has fewer than 6 seeds
    for idx in range(n_seeds, nrows * ncols):
        fig.delaxes(axes_flat[idx])

    # 4. Global Figure Formatting
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        fontsize=11,
        frameon=True,
    )

    plt.tight_layout()
    output_path = f"{save_dir}/qsk_N={N}_multi_seed_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Plot successfully saved to: {output_path}")

# ========================================================================================
# Plot entanglement spectrum (With Lmax)
# ========================================================================================
def plot_spectrum_from_csv_lmax(N, data_dir=None, save_dir=None):
    """Loads extracted entanglement spectrum CSV data and generates a multi-seed

    comparison plot across QSK realizations for all HRBM Lmax variants.
    """
    if data_dir is None:
        data_dir = f"qsk_N={N}"
    if save_dir is None:
        save_dir = data_dir

    csv_path = f"{data_dir}/qsk_N={N}_entanglement_spectra.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV file at: {csv_path}")

    # 1. Load CSV data and find unique Lmax values
    df = pd.read_csv(csv_path)

    qsk_seeds = df["seed_qsk"].unique()
    lmax_values = sorted(df["Lmax"].dropna().astype(int).unique())
    n_seeds = len(qsk_seeds)

    # 2. Setup 3x2 Subplots Grid
    nrows, ncols = 3, 2
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 3.8 * nrows), sharey=True
    )
    axes_flat = axes.flatten()

    handles = []

    # Generate distinct colors for each HRBM Lmax curve
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(lmax_values)))

    # 3. Iterate over QSK disorder seeds
    for idx, seed_qsk in enumerate(qsk_seeds[: nrows * ncols]):
        ax = axes_flat[idx]
        df_seed = df[df["seed_qsk"] == seed_qsk]

        # Use the first Lmax slice for Exact GS and RBM to avoid row duplication
        first_lmax = lmax_values[0]
        df_first_lmax = df_seed[df_seed["Lmax"] == first_lmax]

        rbm_grouped = df_first_lmax.groupby("schmidt_index").agg(
            exact_lambda=("exact_lambda", "first"),
            rbm_mean=("rbm_lambda", "mean"),
            rbm_std=("rbm_lambda", "std"),
        )

        indices = rbm_grouped.index.values

        # Plot Exact GS
        (line_exact,) = ax.semilogy(
            indices,
            rbm_grouped["exact_lambda"],
            "k-",
            label="Exact GS",
            linewidth=2.0,
        )

        # Plot RBM Mean + Shaded Error Band
        (line_rbm,) = ax.semilogy(
            indices,
            rbm_grouped["rbm_mean"],
            "r--",
            label=r"RBM (Euclidean)",
            alpha=0.85,
            linewidth=1.5,
        )
        ax.fill_between(
            indices,
            np.clip(
                rbm_grouped["rbm_mean"] - rbm_grouped["rbm_std"],
                1e-15,
                None,
            ),
            rbm_grouped["rbm_mean"] + rbm_grouped["rbm_std"],
            color="red",
            alpha=0.15,
        )

        seed_handles = [line_exact, line_rbm]

        # Plot HRBM curves for each distinct Lmax
        for l_idx, lmax in enumerate(lmax_values):
            df_lmax = df_seed[df_seed["Lmax"] == lmax]
            hrbm_grouped = df_lmax.groupby("schmidt_index").agg(
                hrbm_mean=("hrbm_lambda", "mean"),
                hrbm_std=("hrbm_lambda", "std"),
            )

            color = colors[l_idx]
            (line_hrbm,) = ax.semilogy(
                indices,
                hrbm_grouped["hrbm_mean"],
                "-.",
                color=color,
                label=rf"HRBM ($L_\mathrm{{max}}={lmax}$)",
                alpha=0.9,
                linewidth=1.5,
            )
            ax.fill_between(
                indices,
                np.clip(
                    hrbm_grouped["hrbm_mean"] - hrbm_grouped["hrbm_std"],
                    1e-15,
                    None,
                ),
                hrbm_grouped["hrbm_mean"] + hrbm_grouped["hrbm_std"],
                color=color,
                alpha=0.12,
            )

            seed_handles.append(line_hrbm)

        if idx == 0:
            handles = seed_handles

        # Apply y-axis label to left-column subplots
        if idx % ncols == 0:
            ax.set_ylabel(r"Eigenvalue $\lambda_i$", fontsize=11)

        ax.set_title(f"QSK Realization (Seed {seed_qsk})", fontsize=11)
        ax.set_xlabel("Schmidt Rank Index $i$", fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.35)

    # Clean up empty subplots if CSV has fewer than 6 seeds
    for idx in range(n_seeds, nrows * ncols):
        fig.delaxes(axes_flat[idx])

    # 4. Global Figure Formatting
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=min(len(handles), 4),
        fontsize=10,
        frameon=True,
    )

    plt.tight_layout()
    output_path = f"{save_dir}/qsk_N={N}_lmax_multi_seed_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Plot successfully saved to: {output_path}")
