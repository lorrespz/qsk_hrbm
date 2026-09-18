import os
import netket as nk
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def generate_symmetric_matrix(N, seed=1234):
    """Generates symmetric Gaussian disorder matrix matching original sk.py."""
    np.random.seed(seed)
    random_numbers = np.random.normal(0, np.sqrt(1 / N), size=(N, N))
    return (random_numbers + random_numbers.T) / 2


def generate_hamiltonian_sk_scipy(N, seed, Gamma=1.0):
    """Generates SK Hamiltonian directly as a SciPy CSR matrix on CPU RAM."""
    V = generate_symmetric_matrix(N, seed=seed)
    dim = 2**N

    # 1. Fast vectorised construction of diagonal Z_i Z_j terms
    spins = np.ones((dim, N), dtype=np.int8)
    for i in range(N):
        mask = 1 << (N - 1 - i)
        spins[:, i] = np.where((np.arange(dim) & mask) == 0, 1, -1)

    diag_H = np.zeros(dim, dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            if V[i, j] != 0:
                diag_H += V[i, j] * (spins[:, i] * spins[:, j])

    del spins
    sp_h = sp.diags(diag_H, format="csr", dtype=np.float64)
    del diag_H

    # 2. Add Transverse Field (-\Gamma \sum X_i) via Kronecker products
    sx = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    eye = sp.eye(2, format="csr", dtype=np.float64)

    for i in range(N):
        ops = [eye] * N
        ops[i] = sx
        X_i = ops[0]
        for op in ops[1:]:
            X_i = sp.kron(X_i, op, format="csr")
        sp_h = sp_h - Gamma * X_i

    return sp_h


def generate_hamiltonian_sk(N, seed, Gamma=1.0):
    """Generates the NetKet QSK Hamiltonian (used for NQS sampling on GPU)."""
    hi = nk.hilbert.Spin(s=1 / 2, N=N)
    V = generate_symmetric_matrix(N, seed=seed)

    H = nk.operator.LocalOperator(hi)

    # Transverse field term
    for i in range(N):
        H = H - Gamma * nk.operator.spin.sigmax(hi, i)

    # Interaction terms J_ij Z_i Z_j
    for i in range(N):
        for j in range(i + 1, N):
            if V[i, j] != 0:
                H = H + V[i, j] * (
                    nk.operator.spin.sigmaz(hi, i)
                    @ nk.operator.spin.sigmaz(hi, j)
                )

    return hi, H


def generate_state_sk(N, seed, Gamma=1.0, path=None):
    """Computes exact ground state on CPU system RAM using SciPy."""
    if path:
        filename = f"{path}/sk_ground_state_N={N}_seed={seed}.npy"
        print(f"Loading cached exact ground state from {filename}...")
        return None, np.load(filename)
    else:
        filename = f"sk_ground_state_N={N}_seed={seed}.npy"
        if os.path.exists(filename):
            print(f"Loading cached exact ground state from {filename}...")
            return None, np.load(filename)

    print(f"Generating N={N} exact ground state on CPU RAM...")
    sp_h = generate_hamiltonian_sk_scipy(N, seed, Gamma=Gamma)

    print(
        f"Sparse matrix constructed ({2**N}x{2**N}). Diagonalizing via SciPy ARPACK..."
    )
    eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
    ground_state_vec = eig_vecs[:, 0]

    np.save(filename, ground_state_vec)
    print(f"Exact GS saved to {filename}. Energy = {eig_vals[0]:.6f}")

    return eig_vals[0], ground_state_vec
