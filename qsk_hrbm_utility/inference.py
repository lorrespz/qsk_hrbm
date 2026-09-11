import os
import flax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import netket as nk
import numpy as np
import pandas as pd
from hrbm import *
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
    """Dynamically searches Kaggle directory patterns to locate .mpack files."""
    # Auto-detect Kaggle workspace root if base_dir is default '.'
    if base_dir == "." and os.path.exists("/kaggle/working/qsk_hrbm"):
        base_dir = "/kaggle/working/qsk_hrbm"

    candidate_paths = [
        # Exact path structure: N=22/qsk_N=22/seed_qsk=1234/seed=101
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
            base_dir,
            f"qsk_N={N}",
            f"seed_qsk={seed_qsk}_seed_nqs={seed_nqs}",
        ),
        os.path.join(
            base_dir,
            f"qsk_N={N}",
            f"qsk_seed={seed_qsk}_nqs_seed={seed_nqs}",
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


def calc_exact_ee(N, seed_qsk, alpha):
    hi, H = generate_hamiltonian_sk(N=N, seed=seed_qsk)

    if generate_hamiltonian_sk_scipy is not None:
        sp_h = generate_hamiltonian_sk_scipy(N, seed_qsk)
    else:
        sp_h = H.to_sparse()

    _, ground_state_vec = generate_state_sk(N=N, seed=seed_qsk)
    exact_E = (ground_state_vec.conj().T @ sp_h @ ground_state_vec).real
    print(f"Exact ED Ground State Energy: {exact_E:.6f}")

    print("Exact ground state wavefunction")
    exact_s2, exact_svn, exact_spec = calculate_entanglement(
        ground_state_vec, N
    )
    return hi, exact_s2, exact_svn, exact_spec


def define_model_load_vstate(N, hi, seed_qsk, alpha, seed_nqs, base_dir="."):
    chkpt_path = find_checkpoint_dir(base_dir, N, seed_qsk, seed_nqs)

    ############################## RBM ##################################
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
    print("RBM Model parameters successfully loaded!")

    ############################## HRBM ##################################
    k = 1.0
    Lmax = 10
    model_hrbm = HyperbolicRBMd_Coupled(alpha=alpha, k=k, Lmax=Lmax)
    vstate_hrbm = nk.vqs.MCState(
        sampler=nk.sampler.MetropolisLocal(hi),
        model=model_hrbm,
        n_samples=1008,
    )

    hrbm_file = os.path.join(
        chkpt_path, f"hrbm_alpha={alpha}_ML_final_params.mpack"
    )
    if not os.path.exists(hrbm_file):
        hrbm_file = os.path.join(
            chkpt_path, "hrbm_alpha=1_ML_final_params.mpack"
        )

    with open(hrbm_file, "rb") as f:
        vstate_hrbm.parameters = flax.serialization.from_bytes(
            vstate_hrbm.parameters, f.read()
        )
    print(f"HRBM Model parameters successfully loaded!")
    print(
        "=============================================================================="
    )
    return vstate_rbm, vstate_hrbm


# ========================================================================================
# Energy properties
# ========================================================================================
def compute_exact_vstate_metrics(vstate, H_sparse, exact_E):
    """Exact matrix-vector evaluation for a variational state vector."""
    psi = np.array(vstate.to_array())
    psi = psi / np.linalg.norm(psi)

    H_psi = H_sparse @ psi
    eval_E = float(np.real(psi.conj().T @ H_psi))

    H2_psi = H_sparse @ H_psi
    eval_E2 = float(np.real(psi.conj().T @ H2_psi))

    variance = max(0.0, eval_E2 - eval_E**2)
    rel_error = float(np.abs(eval_E - exact_E) / np.abs(exact_E))

    return eval_E, rel_error, variance


def evaluate_and_export_energies(
    N, alpha, seed_qsk_list, seed_nqs_list, base_dir=".", save_dir=None
):
    """Evaluates exact ground state energy metrics for loaded checkpoints and saves results to CSV."""
    if save_dir is None:
        save_dir = f"qsk_N={N}"
    os.makedirs(save_dir, exist_ok=True)

    energy_rows = []

    for seed_qsk in seed_qsk_list:
        print(
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        )
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
            print(
                "*******************************************************************************"
            )
            print(f"Current NQS = {seed_nqs}")
            print(
                "*******************************************************************************"
            )
            rbm_vstate, hrbm_vstate = define_model_load_vstate(
                N, hi, seed_qsk, alpha, seed_nqs, base_dir=base_dir
            )

            r_E, r_err, r_var = compute_exact_vstate_metrics(
                rbm_vstate, H_sparse, exact_E
            )
            energy_rows.append(
                {
                    "N": N,
                    "seed_qsk": seed_qsk,
                    "seed_nqs": seed_nqs,
                    "model": "RBM (Euclidean)",
                    "energy": r_E,
                    "rel_energy_error": r_err,
                    "energy_variance": r_var,
                }
            )

            h_E, h_err, h_var = compute_exact_vstate_metrics(
                hrbm_vstate, H_sparse, exact_E
            )
            energy_rows.append(
                {
                    "N": N,
                    "seed_qsk": seed_qsk,
                    "seed_nqs": seed_nqs,
                    "model": "HRBM (Hyperbolic)",
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
    N, alpha, seed_qsk_list, seed_nqs_list, base_dir=".", save_dir=None
):
    if save_dir is None:
        save_dir = f"qsk_N={N}"
    os.makedirs(save_dir, exist_ok=True)

    metrics_rows = []
    spectra_rows = []

    for seed_qsk in seed_qsk_list:
        print(
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        )
        print(f"Current QSK = {seed_qsk}")
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
            print(
                "*******************************************************************************"
            )
            print(f"Current NQS = {seed_nqs}")
            print(
                "*******************************************************************************"
            )
            rbm_vstate, hrbm_vstate = define_model_load_vstate(
                N, hi, seed_qsk, alpha, seed_nqs, base_dir=base_dir
            )

            r_s2, r_svn, r_spec = calculate_entanglement(rbm_vstate, N)
            r_sorted = np.sort(r_spec)[::-1]

            metrics_rows.append(
                {
                    "N": N,
                    "seed_qsk": seed_qsk,
                    "seed_nqs": seed_nqs,
                    "model": "RBM (Euclidean)",
                    "S_2": r_s2,
                    "S_vN": r_svn,
                }
            )

            h_s2, h_svn, h_spec = calculate_entanglement(hrbm_vstate, N)
            h_sorted = np.sort(h_spec)[::-1]

            metrics_rows.append(
                {
                    "N": N,
                    "seed_qsk": seed_qsk,
                    "seed_nqs": seed_nqs,
                    "model": "HRBM (Hyperbolic)",
                    "S_2": h_s2,
                    "S_vN": h_svn,
                }
            )

            for idx in range(len(exact_sorted)):
                spectra_rows.append(
                    {
                        "N": N,
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
