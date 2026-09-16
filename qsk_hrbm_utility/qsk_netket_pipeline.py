import time
import math
import jax
import copy
import os
import json
import flax
import flax.linen as nn
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax
import matplotlib.pyplot as plt
from netket_callbacks import *
from scipy.sparse.linalg import eigsh


def set_seed(seed):
    """Generates deterministic JAX PRNG keys for NetKet pipeline."""
    key = jax.random.PRNGKey(seed)
    key, key_model, key_sampler, key_vstate = jax.random.split(key, 4)
    return key_model, key_sampler, key_vstate

########################################################################################################
# Netket pipeline
########################################################################################################
def create_optimizer(learning_rate):
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=learning_rate)
    )
      
def netket_pipeline(N, model, sampler, op, vstate, optimizer, iter_num, exact_E, holomorphic=True,
                    log_filename="qsk_rbm_results", 
                    save_params=True, patience=40, lr_factor=0.5): 
    #  Automatically create output directory if it doesn't exist
    log_dir = os.path.dirname(log_filename)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    #  Create the Variational State (MCState)
    # Use the SR (Stochastic Reconfiguration) preconditioner with holomorphic=True
    if holomorphic:
        preconditioner = nk.optimizer.SR(diag_shift=0.01, holomorphic=True)
    else: 
        preconditioner = nk.optimizer.SR(diag_shift=0.05, holomorphic=False)
    gs = nk.driver.VMC(op, optimizer, variational_state=vstate, preconditioner=preconditioner)
    
    #Setup Loggers: RuntimeLog (in-memory for plotting) + JsonLog (disk file for saving)
    runtime_log = nk.logging.RuntimeLog()
    json_log = nk.logging.JsonLog(
        log_filename, 
        write_every=10, 
        save_params_every=10,  # Saves variational parameters every 50 steps
        save_params=save_params
    )
    
    # Combine loggers into a list
    loggers = [runtime_log, json_log]

    # 1. Instantiate callbacks
    text_cb = text_callback
    lr_cb = ReduceLROnPlateauCallback(patience=30, factor=0.5, min_lr=1e-6)
    es_cb = EarlyStoppingWithCheckpointing(patience=100, min_delta=1e-4)    
    
    # 2. Master callback
    def master_callback(step, log_data, driver):
      c1 = text_cb(step, log_data, driver)
      c2 = lr_cb(step, log_data, driver)
      c3 = es_cb(step, log_data, driver)
      return c1 and c2 and c3

    # 3. Run training driver
    gs.run(n_iter=iter_num, out=loggers, show_progress=True, callback=master_callback)
    
    # 4. Restore best parameters before computing final expectation values & saving
    es_cb.restore_best(vstate, exact_E)

    # 5. Extract and print final energy statistics
    final_energy = runtime_log.data['Energy']['Mean'][-1]
    stats = vstate.expect(op)   
    
    print("\n" + "="*40)
    print(f"Mean Energy:        {stats.mean.real:.6f}")
    print(f"Error Bar:          ± {stats.error_of_mean:.6f}")
    print(f"Variance:           {stats.variance:.6f}")
    print(f"Final VMC Energy:   {final_energy.real:.6f}")

    if 'Ee' in globals():
        print(f"Exact Energy (ED):  {Ee:.6f}")
    print("="*40)
    
    # 6. Save final model parameters explicitly to a MsgPack file
    param_file = f"{log_filename}_final_params.mpack"
    with open(param_file, "wb") as f:
        f.write(flax.serialization.to_bytes(vstate.parameters))
    print(f"Final model weights saved to: {param_file}")

    # 7. Convergence Plotting
    if 'Energy' in runtime_log.data:
        history = runtime_log.data
        plt.figure(figsize=(8, 5))
        plt.plot(history['Energy']['iters'], history['Energy']['Mean'].real, label='VMC Energy')

        plt.axhline(y=exact_E, color='r', linestyle='--', label='ED Target')            
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.legend()
        plt.title(f'QSK NQS Convergence')
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("Training finished, but no data was logged.")


########################################################################################################
# Load trained mpack files
########################################################################################################
def load_files(file_path):
    # Load the JSON data
    with open(file_path, "r") as f:
        log_data = json.load(f)
    
    # 2. Extract iteration numbers
    iters = log_data["Energy"]["iters"]
    raw_energies = log_data["Energy"]["Mean"]
    raw_variances = log_data["Energy"]["Variance"]
    return iters, raw_energies, raw_variances
########################################################################################################
# Calculate Rényi Entropies and von Neumann Entropy
########################################################################################################
def calculate_renyi(evs, alpha):
    if alpha == 1.0:
        # Limit alpha -> 1 is the von Neumann entropy: S = -Tr(rho ln rho)
        return -jnp.sum(evs * jnp.log(evs))
    else:
        # S_alpha = 1 / (1 - alpha) * ln(Tr(rho^alpha))
        return (1.0 / (1.0 - alpha)) * jnp.log(jnp.sum(evs**alpha))

def calculate_entanglement(vstate, N):
    # 1. Extract the full dense wave function vector from the trained NetKet vstate    
    # Check if input is already a raw numpy/jax array (Exact Diagonalization result)
    if isinstance(vstate, (np.ndarray, jnp.ndarray)):
        # Shape will be (2^N,) -> (1024,) for N=10
        psi = jnp.asarray(vstate)
    
    # Otherwise, assume it's a NetKet variational state with a .to_array() method
    elif hasattr(vstate, 'to_array'):
        psi = vstate.to_array()
    else:
        raise TypeError("Input must be either a NetKet vstate or a numpy/jax wave function array.")
    psi = psi / jnp.linalg.norm(psi)  # Ensure proper normalization
    # 2. Reshape into a multi-qubit tensor
    psi_tensor = psi.reshape(tuple([2] * N))
    # 3. Define a bipartition: Subsystem A (first n_A qubits) and Subsystem B (the rest)
    n_A = N//2  
    n_B = N - n_A
    indices_A = list(range(n_A))
    indices_B = list(range(n_A, N))
    # Transpose tensor to group subsystem A and B indices together
    axes = indices_A + indices_B
    psi_matrix = jnp.transpose(psi_tensor, axes).reshape((2**n_A, 2**n_B))
    # 4. Compute the Reduced Density Matrix for Subsystem A: rho_A = psi_matrix @ psi_matrix^dagger
    rho_A = jnp.dot(psi_matrix, jnp.conj(psi_matrix.T))
    # 5. Diagonalize rho_A to get the Entanglement Spectrum (eigenvalues)
    # Eigenvalues are sorted in ascending order by default
    eigenvalues = jnp.linalg.eigvalsh(rho_A)
    # Clean up numerical noise (clip tiny negative values from float precision)
    eigenvalues = jnp.clip(eigenvalues, 1e-15, None)
    eigenvalues = eigenvalues / jnp.sum(eigenvalues)  # Re-normalize trace to 1
    
    s_2 = calculate_renyi(eigenvalues, 2.0)   # 2nd Rényi Entropy
    s_vN = calculate_renyi(eigenvalues, 1.0)  # von Neumann Entropy
    
    print(f"--- Entanglement Metrics (Subsystem A size = {n_A}) ---")
    print(f"2nd Rényi Entropy (S_2): {s_2:.4f}")
    print(f"von Neumann Entropy (S_vN): {s_vN:.4f}")
    print("\nEntanglement Spectrum (Top 10 eigenvalues of rho_A):")
    print(np.sort(eigenvalues)[::-1][:10])  # Print largest eigenvalues descending
    return s_2, s_vN, eigenvalues
