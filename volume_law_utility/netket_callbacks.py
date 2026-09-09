import time
import jax
import os
import flax
import netket as nk
import numpy as np
from lmath_jax import *
########################################################################################################
# Netket pipeline - Callbacks definitions
########################################################################################################

def get_hyperparams(driver):
  """Safely extracts hyperparams dictionary from NetKet driver's internal Optax state."""
  opt_state = getattr(
      driver, "_optimizer_state", getattr(driver, "optimizer_state", None)
  )
  if opt_state is None:
    return None

  # Direct InjectHyperparamsState
  if hasattr(opt_state, "hyperparams"):
    return opt_state.hyperparams

  # Nested Optax state (e.g. inside optax.chain or tuple states)
  if isinstance(opt_state, (tuple, list)):
    for sub_state in opt_state:
      if hasattr(sub_state, "hyperparams"):
        return sub_state.hyperparams
  return None


def extract_lr_float(hp_dict, key):
  """Safely converts a potentially complex JAX/NumPy LR scalar to a Python float."""
  val = hp_dict[key]
  if hasattr(val, "real"):
    return float(val.real)
  return float(np.real(val))


class ReduceLROnPlateauCallback:
  def __init__(self, patience=30, factor=0.5, min_lr=1e-6):
    self.patience = patience
    self.factor = factor
    self.min_lr = min_lr
    self.best_energy = float("inf")
    self.wait = 0

  def __call__(self, step, log_data, driver):
    current_energy = log_data["Energy"].mean.real

    # Track minimum energy improvement
    if current_energy < self.best_energy - 1e-4:
      self.best_energy = current_energy
      self.wait = 0
    else:
      self.wait += 1
      if self.wait >= self.patience:
        hp = get_hyperparams(driver)
        if hp is not None:
          key = (
              "learning_rate"
              if "learning_rate" in hp
              else ("lr" if "lr" in hp else None)
          )
          if key:
            old_lr = extract_lr_float(hp, key)
            if old_lr > self.min_lr:
              new_lr = max(old_lr * self.factor, self.min_lr)
              hp[key] = new_lr  # Update hyperparameter
              print("\n" + "=" * 55)
              print(
                  f" >>> [LR REDUCTION] Step {step:03d}: Energy stalled at"
                  f" {current_energy:.6f}"
              )
              print(f" >>> Lowering LR: {old_lr:.2e} --> {new_lr:.2e}")
              print("=" * 55 + "\n")
        else:
             print(
              "\n[Warning] Optimizer must be wrapped with"
              " optax.inject_hyperparams.\n"
          )
        self.wait = 0
    return True

class EarlyStoppingWithCheckpointing:
  def __init__(self, patience=100, min_delta=1e-4):
    self.patience = patience
    self.min_delta = min_delta
    self.best_energy = float('inf')
    self.best_params = None
    self.best_step = 0
    self.wait = 0

  def __call__(self, step, log_data, driver):
    current_energy = log_data['Energy'].mean.real

    if current_energy < self.best_energy - self.min_delta:
      self.best_energy = current_energy
      self.best_step = step
      # Fix: Access parameters via driver.state
      self.best_params = jax.tree_util.tree_map(
          lambda x: jax.device_get(x), driver.state.parameters
      )
      self.wait = 0
    else:
      self.wait += 1
      if self.wait >= self.patience:
        print('\n' + '=' * 60)
        print(
            f' >>> [EARLY STOPPING] Step {step:03d}: No energy improvement over'
            f' last {self.patience} steps.'
        )
        print(
            f' >>> Restoring best parameters from step {self.best_step:03d}'
            f' (Energy: {self.best_energy:.6f})...'
        )
        print('=' * 60 + '\n')

        # Access parameters via driver.state
        if self.best_params is not None:
          driver.state.parameters = self.best_params

        return False

    return True

  def restore_best(self, vstate, exact_E):
      if self.best_params is not None:
          vstate.parameters = self.best_params
          print(f'\nRestored best model weights from step {self.best_step:03d} (Energy:'
              f' {self.best_energy:.6f})\n')
          error = abs((exact_E - self.best_energy) / exact_E)
          print(f"Relative error   : {error}")

def text_callback(step, log_data, driver):
    if step % 10 == 0:
        energy = log_data["Energy"].mean.real
        variance = log_data["Energy"].variance

        # Extract active learning rate safely
        current_lr = "N/A"
        hp = get_hyperparams(driver)
        if hp is not None:
          key = (
              "learning_rate"
              if "learning_rate" in hp
              else ("lr" if "lr" in hp else None)
          )
          if key:
            current_lr = f"{extract_lr_float(hp, key):.2e}"

        print(
            f"Iteration {step:03d}: Energy = {energy:.6f}, Var = {variance:.4f},"
            f" LR = {current_lr}"
        )
    return True
########################################################################################################
 # Setting chunk
########################################################################################################
def set_balanced_chunk_size(vstate, min_chunk=256, max_chunk=512):
    n_samples = vstate.n_samples

    # Pick largest power-of-two <= max_chunk that cleanly divides n_samples
    chunk = max_chunk
    while n_samples % chunk != 0 and chunk > min_chunk:
        chunk //= 2

    vstate.chunk_size = chunk
    vstate.sample_chunk_size = chunk
    print(f'>>> Configured vstate.chunk_size = {chunk}')
    
def auto_set_chunk_size(vstate, target_mem_ratio=0.4):
    """Dynamically sets vstate.chunk_size based on available GPU VRAM and model parameters."""
    n_samples = vstate.n_samples

    # 1. Query available VRAM on current JAX device
    try:
        gpu_device = jax.devices('gpu')[0]
        mem_stats = gpu_device.memory_stats()
        available_vram = mem_stats['bytes_limit'] - mem_stats.get(
            'bytes_in_use', 0
        )
    except Exception:
        # Fallback to conservative estimate (e.g. 12 GB) if memory stats unavailable
        available_vram = 12 * (1024**3)

    # 2. Estimate parameter size footprint
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(vstate.parameters))
    bytes_per_complex = 16  # complex128

    # Over-allocation safety multiplier accounting for Jacobians & off-diagonal states
    memory_overhead_per_sample = n_params * bytes_per_complex * 40

    # 3. Compute optimal chunk size fitting inside safe VRAM budget
    vram_budget = available_vram * target_mem_ratio
    raw_chunk = int(vram_budget / max(memory_overhead_per_sample, 1))

    # 4. Snap to nearest power-of-two <= total samples
    if raw_chunk < 1:
        chunk_size = 1
    else:
        chunk_size = 2 ** int(math.log2(raw_chunk))
        chunk_size = min(chunk_size, n_samples)

    # Ensure chunk_size divides n_samples cleanly
    while n_samples % chunk_size != 0 and chunk_size > 1:
        chunk_size //= 2

    # 5. Apply chunk settings
    vstate.chunk_size = chunk_size
    vstate.sample_chunk_size = chunk_size

    print(f'>>> [AUTO-TUNE] Configured vstate.chunk_size = {chunk_size} (N_samples'f' = {n_samples})')
    return chunk_size


