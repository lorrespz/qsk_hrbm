import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sk import generate_hamiltonian_sk, generate_state_sk
from qsk_netket_pipeline import *
from hrbm import *

def calc_exact_ee(N, seed_qsk, alpha):
    # 1. Setup SK Hamiltonian & Hilbert space
    hi, H = generate_hamiltonian_sk(N=N, seed=seed_qsk)
    
    # 2. Compute Exact Diagonalization (ED) target energy
    _, ground_state_vec = generate_state_sk(N=N, seed=seed_qsk)
    sp_h = H.to_sparse()
    exact_E = (ground_state_vec.conj().T @ sp_h @ ground_state_vec).real
    print(f"Exact ED Ground State Energy: {exact_E:.6f}")
    
    print('Exact ground state wavefunction')
    # Returns (s2, svn, eigenvalues)
    exact_s2, exact_svn, exact_spec = calculate_entanglement(ground_state_vec, N) 
    return hi, exact_s2, exact_svn, exact_spec

def define_model_load_vstate(N, hi, seed_qsk, alpha, seed_nqs):
    path = f'qsk_N={N}/seed_qsk={seed_qsk}/seed={seed_nqs}'
    ############################## RBM ##################################
    model_rbm = nk.models.RBM(alpha=alpha) 
    vstate_rbm = nk.vqs.MCState(sampler = nk.sampler.MetropolisLocal(hi), 
        model=model_rbm, 
        n_samples=1008
    )
    with open(f'{path}/rbm_alpha=1_ML_final_params.mpack', 'rb') as f:
        vstate_rbm.parameters = flax.serialization.from_bytes(vstate_rbm.parameters, f.read())
    print("RBM Model parameters successfully loaded!")
    print(f"Total RBM parameters: {vstate_rbm.n_parameters}")
    
    ############################## HRBM ##################################
    k=1.0
    Lmax=10
    model_hrbm = HyperbolicRBMd_Coupled(alpha=alpha, k=k, Lmax=Lmax)
    vstate_hrbm= nk.vqs.MCState(
        sampler = nk.sampler.MetropolisLocal(hi), 
        model=model_hrbm, 
        n_samples=1008
    )
    with open(f'{path}/hrbm_alpha=1_ML_final_params.mpack', 'rb') as f:
        vstate_hrbm.parameters = flax.serialization.from_bytes(vstate_hrbm.parameters, f.read())
    print("HRBM Model parameters successfully loaded!")
    print(f"Total HRBM (Lmax={Lmax}) parameters: {vstate_hrbm.n_parameters}")
    print(f'==============================================================================')
    return vstate_rbm, vstate_hrbm

#========================================================================================
# Energy properties
#========================================================================================
def compute_exact_vstate_metrics(vstate, H_sparse, exact_E):
    """Exact matrix-vector evaluation for a variational state vector."""
    # 1. Reconstruct full 2^N state vector and normalize
    psi = np.array(vstate.to_array())
    psi = psi / np.linalg.norm(psi)

    # 2. Compute exact expectation value <psi|H|psi>
    H_psi = H_sparse @ psi
    eval_E = float(np.real(psi.conj().T @ H_psi))

    # 3. Compute exact squared expectation value <psi|H^2|psi>
    H2_psi = H_sparse @ H_psi
    eval_E2 = float(np.real(psi.conj().T @ H2_psi))

    # 4. Variance and relative error
    variance = max(0.0, eval_E2 - eval_E**2)
    rel_error = float(np.abs(eval_E - exact_E) / np.abs(exact_E))

    return eval_E, rel_error, variance


def evaluate_and_export_energies(
    N, alpha, seed_qsk_list, seed_nqs_list, save_dir=None
):
    """Evaluates exact ground state energy metrics for loaded checkpoints and

    saves results to CSV.
    """
    if save_dir is None:
        save_dir = f"qsk_N={N}"
    os.makedirs(save_dir, exist_ok=True)

    energy_rows = []

    for seed_qsk in seed_qsk_list:
        print(f'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print(f'Current QSK = {seed_qsk}')
        hi, H = generate_hamiltonian_sk(N=N, seed=seed_qsk)
        H_sparse = H.to_sparse()

        _, ground_state_vec = generate_state_sk(N=N, seed=seed_qsk)
        exact_E = float(
            (ground_state_vec.conj().T @ H_sparse @ ground_state_vec).real
        )

        # Record Exact Ground State baseline
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
            print(f'*******************************************************************************')
            print(f'Current NQS = {seed_nqs}')
            print(f'*******************************************************************************')
            rbm_vstate, hrbm_vstate = define_model_load_vstate(N, hi, seed_qsk=seed_qsk, alpha=alpha,
                                                             seed_nqs=seed_nqs)

            # Evaluate RBM
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

            # Evaluate HRBM
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

    # Convert to DataFrame
    df_energy = pd.DataFrame(energy_rows)

    # Save to CSV
    csv_path = f"{save_dir}/qsk_N={N}_energy_metrics.csv"
    df_energy.to_csv(csv_path, index=False)
    print(f"Energy metrics saved successfully to: {csv_path}")

    # Generate Statistical Summary DataFrame (Mean ± Std)
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
#========================================================================================
# Entanglement properties
#========================================================================================
def entanglement_extraction(N, alpha, seed_qsk_list, seed_nqs_list, save_dir=None):
    if save_dir is None:
        save_dir = f"qsk_N={N}"
    os.makedirs(save_dir, exist_ok=True)

    metrics_rows = []
    spectra_rows = []

    for seed_qsk in seed_qsk_list:
        # Unpack s2, svn, and spec directly from calc_exact_ee
        print(f'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print(f'Current QSK = {seed_qsk}')
        hi, exact_s2, exact_svn, exact_spec = calc_exact_ee(N, seed_qsk, alpha)
        exact_sorted = np.sort(exact_spec)[::-1]
        print(f'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        # Record Exact GS scalar metrics
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
            print(f'*******************************************************************************')
            print(f'Current NQS = {seed_nqs}')
            print(f'*******************************************************************************')
            rbm_vstate, hrbm_vstate = define_model_load_vstate(
                N, hi, seed_qsk=seed_qsk, alpha=alpha, seed_nqs=seed_nqs)

            # 1. Compute Entanglement for RBM
            print(f'RBM ENTANGLEMENT PROPERTIES')
            print(f'==============================================================================')
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

            # 2. Compute Entanglement for HRBM
            print(f'==============================================================================')
            print(f'HRBM ENTANGLEMENT PROPERTIES')
            print(f'==============================================================================')
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

            # 3. Store detailed spectrum data for each mode index
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

    # Convert to DataFrames
    df_metrics = pd.DataFrame(metrics_rows)
    df_spectra = pd.DataFrame(spectra_rows)

    # Save to CSV
    metrics_csv_path = f"{save_dir}/qsk_N={N}_entanglement_metrics.csv"
    spectra_csv_path = f"{save_dir}/qsk_N={N}_entanglement_spectra.csv"

    df_metrics.to_csv(metrics_csv_path, index=False)
    df_spectra.to_csv(spectra_csv_path, index=False)

    print(f"Metrics saved to: {metrics_csv_path}")
    print(f"Spectra saved to: {spectra_csv_path}")

    # Generate Statistical Summary DataFrame (Mean ± Std)
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

def plot_qsk_seeds_separately(
    N, alpha, seed_qsk_list, seed_nqs_list, save_dir=f"qsk_N=20"
):
    fig, axes = plt.subplots(
        1,
        len(seed_qsk_list),
        figsize=(5 * len(seed_qsk_list), 4.2),
        sharey=True,
    )

    for idx, seed_qsk in enumerate(seed_qsk_list):

        ax = axes[idx] if len(seed_qsk_list) > 1 else axes
        print(f'Current QSK = {seed_qsk}')
        hi, _, _, exact_spec = calc_exact_ee(N, seed_qsk, alpha)
        exact_sorted = np.sort(exact_spec)[::-1]

        rbm_spectra_seed, hrbm_spectra_seed = [], []

        for seed_nqs in seed_nqs_list:
            rbm_vstate, hrbm_vstate = define_model_load_vstate(N, hi, seed_qsk=seed_qsk, 
                alpha=alpha, seed_nqs=seed_nqs)
            _, _, r_spec = calculate_entanglement(rbm_vstate, N)
            _, _, h_spec = calculate_entanglement(hrbm_vstate, N)

            rbm_spectra_seed.append(np.sort(r_spec)[::-1])
            hrbm_spectra_seed.append(np.sort(h_spec)[::-1])

        rbm_arr = np.array(rbm_spectra_seed)
        hrbm_arr = np.array(hrbm_spectra_seed)

        rbm_mean, rbm_std = np.mean(rbm_arr, axis=0), np.std(rbm_arr, axis=0)
        hrbm_mean, hrbm_std = np.mean(hrbm_arr, axis=0), np.std(hrbm_arr, axis=0)

        indices = np.arange(1, len(exact_sorted) + 1)

        (line1,) = ax.semilogy(indices, exact_sorted, "k-", label="Exact GS", linewidth=2.0)
        (line2,) = ax.semilogy(indices, rbm_mean, "r--", label=r"RBM (Euclidean) $\pm 1\sigma$", alpha=0.85, linewidth=1.5)
        ax.fill_between(indices, np.clip(rbm_mean - rbm_std, 1e-15, None), rbm_mean + rbm_std, color="red", alpha=0.15)

        (line3,) = ax.semilogy(indices, hrbm_mean, "b-.", label=r"HRBM (Hyperbolic) $\pm 1\sigma$", alpha=0.85, linewidth=1.5)
        ax.fill_between(indices, np.clip(hrbm_mean - hrbm_std, 1e-15, None), hrbm_mean + hrbm_std, color="blue", alpha=0.15)

        ax.set_title(f"QSK Realization (Seed {seed_qsk})", fontsize=11)
        ax.set_xlabel("Schmidt Rank Index $i$", fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.35)

        if idx == 0:
            ax.set_ylabel(r"Eigenvalue $\lambda_i$", fontsize=11)

    fig.legend(
        handles=[line1, line2, line3],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        fontsize=11,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(f"{save_dir}/qsk_N={N}_multi_seed_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_spectrum_from_csv(N, data_dir=None, save_dir=None):
    """Loads extracted entanglement spectrum CSV data and generates a multi-seed

    comparison plot across QSK realizations.
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

    # 2. Setup Subplots
    fig, axes = plt.subplots(
        1, n_seeds, figsize=(5 * n_seeds, 4.2), sharey=True
    )
    if n_seeds == 1:
        axes = [axes]

    handles = []

    # 3. Iterate over QSK disorder seeds
    for idx, seed_qsk in enumerate(qsk_seeds):
        ax = axes[idx]
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
            ax.set_ylabel(r"Eigenvalue $\lambda_i$", fontsize=11)

        ax.set_title(f"QSK Realization (Seed {seed_qsk})", fontsize=11)
        ax.set_xlabel("Schmidt Rank Index $i$", fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.35)

    # 4. Global Figure Formatting
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        fontsize=11,
        frameon=True,
    )

    plt.tight_layout()
    output_path = f"{save_dir}/qsk_N={N}_multi_seed_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Plot successfully saved to: {output_path}")
