import time
import jax
import os
import flax
import flax.linen as nn
import jax.numpy as jnp
import netket as nk
import numpy as np
from lmath_jax import *

def stable_complex_log_2cosh(x):
    return jnp.where(
        jnp.real(x) >= 0,
        x + jnp.log1p(jnp.exp(-2.0 * x)),
        -x + jnp.log1p(jnp.exp(2.0 * x))
    )

#####################################################################################
# Euclidean RBM (matching NetKet built-in construction)
#####################################################################################
class ExplicitEuclideanRBM(nn.Module):
    alpha: float = 1.0
    param_dtype: object = jnp.complex128

    @nn.compact
    def __call__(self, x):
        # 1. Define dimensions
        N = x.shape[-1]
        M = int(self.alpha * N)

        # 2. Define standard RBM parameters
        a = self.param("visible_bias", nn.initializers.zeros, (N,), self.param_dtype)
        b = self.param("hidden_bias", nn.initializers.zeros, (M,), self.param_dtype)
        
        # In Euclidean RBM, the interaction matrix W is a direct, dense parameter matrix
        init_fn = nn.initializers.normal(stddev=0.05)
        W = self.param("W", init_fn, (N, M), self.param_dtype)

        # 3. Visible Field component
        visible_term = jnp.dot(x, a)

        # 4. Hidden Field component (Matrix multiplication + bias)
        theta = b + jnp.dot(x, W)

        # 5. Apply the RBM activation function
        hidden_term = jnp.sum(stable_complex_log_2cosh(theta), axis=-1)

        # 6. Sum them up for the final log-wavefunction
        return visible_term + hidden_term

###################################################################################################
# HRBM construction
###################################################################################################
class HyperbolicRBM(nn.Module):
    alpha: float = 4.0         
    k: float = 1.0             
    Lmax: float = 50.0          
    param_dtype: object = jnp.complex128  

    @nn.compact
    def __call__(self, x):
        x_real = x.astype(jnp.float64)
        N = x.shape[-1]
        M = int(self.alpha * N)
        
        # 1. Initialize Complex Tangent-Space Parameters
        W_tangent = self.param("W_tangent", nn.initializers.normal(stddev=0.01), (M, N), self.param_dtype)
        a_tangent = self.param("visible_bias_tangent", nn.initializers.normal(stddev=0.05), (N,), self.param_dtype)
        b_tangent = self.param("hidden_bias_tangent", nn.initializers.normal(stddev=0.01), (M,), self.param_dtype)
        
        W_re, W_im = jnp.real(W_tangent), jnp.imag(W_tangent)
        a_re, a_im = jnp.real(a_tangent), jnp.imag(a_tangent)
        b_re, b_im = jnp.real(b_tangent), jnp.imag(b_tangent)
        
        # 2. Shared input mapping on the real Lorentz hyperboloid
        Z_x = LorentzMathJAX.expmap0(x_real, k=self.k) 
        
        # 3. Visible Term (Complex-valued linear part)
        visible_term = jnp.dot(x_real, a_re) + 1j * jnp.dot(x_real, a_im)
        
        # ==========================================
        # Branch 1: Real Lorentz Manifold (Amplitude Geometry)
        # ==========================================
        Z_b_re = LorentzMathJAX.expmap0(b_re, k=self.k) 
        Z_xw_re = LorentzMathJAX.mobius_matvec_clamped(W_re, Z_x, k=self.k)
        Z_hid_re = LorentzMathJAX.mobius_add_clamped(Z_b_re, Z_xw_re, k=self.k)
        Z_hid_re = LorentzMathJAX.apply_spatial_clamp(Z_hid_re, k=self.k, spatial_clamp=self.Lmax)
        theta_re = LorentzMathJAX.logmap0(Z_hid_re, k=self.k)
        
        # ==========================================
        # Branch 2: Imaginary Lorentz Manifold (Phase Geometry)
        # ==========================================
        Z_b_im = LorentzMathJAX.expmap0(b_im, k=self.k) 
        Z_xw_im = LorentzMathJAX.mobius_matvec_clamped(W_im, Z_x, k=self.k)
        Z_hid_im = LorentzMathJAX.mobius_add_clamped(Z_b_im, Z_xw_im, k=self.k)
        Z_hid_im = LorentzMathJAX.apply_spatial_clamp(Z_hid_im, k=self.k, spatial_clamp=self.Lmax)
        theta_im = LorentzMathJAX.logmap0(Z_hid_im, k=self.k)
        
        # ==========================================
        # 4. Complex Non-Linear Cross-Coupling
        # ==========================================
        # Reconstruct the complex pre-activation vector from the two independent manifolds
        theta_complex = theta_re + 1j * theta_im
        
        # Pass through the exact same complex activation used in the Euclidean baseline
        hidden_term = jnp.sum(stable_complex_log_2cosh(theta_complex), axis=-1)

        return visible_term + hidden_term
