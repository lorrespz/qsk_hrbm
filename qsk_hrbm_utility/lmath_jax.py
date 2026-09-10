import jax
import jax.numpy as jnp

EXP_MAX_NORM = 100.0
EPS = 1e-7

class LorentzMathJAX:
    """
    JAX-compatible and complex-safe implementation of the HyperCore lmath module
    with correct N-dimensional to (1+N)-dimensional manifold mapping.
    """
    
    @staticmethod
    def arcosh(x, eps=EPS):
        """Safe arcosh ensuring real-domain casting for clipping."""
        x_real = jnp.real(x)
        x_safe = jnp.clip(x_real, a_min=1.0 + eps)
        z = jnp.sqrt(jnp.clip(x_safe**2 - 1.0, a_min=1e-9))
        return jnp.log(x_safe + z)

    @staticmethod
    def inner(u, v, keepdim=False):
        """Minkowski inner product."""
        res = -u[..., 0] * v[..., 0] + jnp.sum(u[..., 1:] * v[..., 1:], axis=-1)
        if keepdim:
            return jnp.expand_dims(res, axis=-1)
        return res

    @staticmethod
    def inner0(v, k=1.0, keepdim=False):
        """Minkowski inner product with the zero vector."""
        res = -v[..., 0] * jnp.sqrt(k)
        if keepdim:
            return jnp.expand_dims(res, axis=-1)
        return res

    @staticmethod
    def norm(u, keepdim=False):
        """Compute vector norm on the tangent space safely."""
        val = LorentzMathJAX.inner(u, u, keepdim=keepdim)
        safe_val = jnp.clip(jnp.real(val), a_min=0.0)
        return jnp.sqrt(safe_val)

    @staticmethod
    def dist(x, y, k=1.0, keepdim=False, eps=EPS):
        """Safe geodesic distance on the Hyperboloid."""
        inner_prod = -LorentzMathJAX.inner(x, y, keepdim=keepdim)
        res = jnp.clip(jnp.real(inner_prod / k), a_min=1.0 + eps)
        return LorentzMathJAX.arcosh(res, eps=eps)

    @staticmethod
    def dist0(x, k=1.0, keepdim=False, eps=EPS):
        """Safe geodesic distance to the origin."""
        d = -LorentzMathJAX.inner0(x, k=k, keepdim=keepdim)
        res = jnp.clip(jnp.real(d / k), a_min=1.0 + eps)
        return LorentzMathJAX.arcosh(res, eps=eps)

    @staticmethod
    def expmap(x, u, k=1.0, eps=1e-10):
        """Safe exponential map from point x"""
        nomin = LorentzMathJAX.norm(u, keepdim=True)
        u_normalized = jnp.where(nomin > eps, u / (nomin + eps), jnp.zeros_like(u))
        nomin_clamped = jnp.clip(nomin, a_max=EXP_MAX_NORM)
        sqrt_c = jnp.sqrt(k)
        
        p = jnp.cosh(nomin_clamped / sqrt_c) * x + \
            jnp.sinh(nomin_clamped / sqrt_c) * u_normalized * sqrt_c
        return p

    @staticmethod
    def expmap0(u, k=1.0, norm_control=False, eps=1e-10):
        """
        Safe exponential map from the origin.
        u: Spatial Euclidean vector of shape (..., N).
        Returns a Lorentz vector of shape (..., 1 + N).
        """
        norm_u = jnp.clip(jnp.real(jnp.linalg.norm(u, axis=-1, keepdims=True)), a_max=100.0)
        
        denom = norm_u / jnp.sqrt(k)
        if norm_control:
            denom = jnp.clip(denom, a_max=1.0)
            
        sqrt_k = jnp.sqrt(k)
        
        # Time component (0-th component)
        l_v = sqrt_k * jnp.cosh(denom)
        
        # Spatial components (1 to N)
        sync_factor = jnp.where(
            norm_u > eps, 
            jnp.sinh(denom) / (norm_u + eps), 
            1.0 / sqrt_k
        )
        r_v = sqrt_k * sync_factor * u
        
        return jnp.concatenate([l_v, r_v], axis=-1)

    @staticmethod
    def logmap(x, y, k=1.0, eps=EPS):
        """Safe logarithmic map between x and y."""
        dist_ = LorentzMathJAX.dist(x, y, k=k, keepdim=True, eps=eps)
        nomin = y + LorentzMathJAX.inner(x, y, keepdim=True) * x / k
        denom = LorentzMathJAX.norm(nomin, keepdim=True)
        
        y_tan = jnp.where(
            dist_ > eps,
            dist_ * nomin / (denom + eps),
            jnp.zeros_like(nomin)
        )
        return y_tan

    @staticmethod
    def logmap0(y, k=1.0, eps=EPS):
        """
        Safe logarithmic map from the origin.
        y: Lorentz vector of shape (..., 1 + N).
        Returns spatial tangent vector of shape (..., N).
        """
        dist_ = LorentzMathJAX.dist0(y, k=k, keepdim=True, eps=eps)
        y_spatial = y[..., 1:]
        spatial_norm = jnp.linalg.norm(y_spatial, axis=-1, keepdims=True)
        
        factor = jnp.where(
            spatial_norm > eps,
            dist_ / (spatial_norm + eps),
            0.0
        )
        return y_spatial * factor

    @staticmethod
    def ptransp(x, y, v, k=1.0):
        """Transport vector v from x to y."""
        K = 1.0 / k
        yv = LorentzMathJAX.inner(y, v, keepdim=True)
        xy = LorentzMathJAX.inner(x, y, keepdim=True)
        denom = jnp.clip(jnp.real(1.0 - K * xy), a_min=1e-6)
        _frac = (K * yv) / denom
        return v + _frac * (x + y)

    @staticmethod
    def ptransp0(y, v, k=1.0):
        """Transport vector v from the origin to y."""
        zeros_head = jnp.ones_like(v[..., 0:1]) * jnp.sqrt(k)
        zeros_tail = jnp.zeros_like(v[..., 1:])
        zeros = jnp.concatenate([zeros_head, zeros_tail], axis=-1)
        return LorentzMathJAX.ptransp(zeros, y, v, k=k)

    @staticmethod
    def mobius_add_clamped(x, y, k=1.0, max_v_norm=150.0):
        """Safe mobius addition."""
        u_spatial = LorentzMathJAX.logmap0(y, k=k)
        zeros_head = jnp.zeros_like(u_spatial[..., 0:1])
        u_ambient = jnp.concatenate([zeros_head, u_spatial], axis=-1)
        
        v = LorentzMathJAX.ptransp0(x, u_ambient, k=k)
        
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        scale = jnp.where(v_norm > max_v_norm, max_v_norm / (v_norm + 1e-10), 1.0)
        v_clamped = v * scale
        return LorentzMathJAX.expmap(x, v_clamped, k=k)

    @staticmethod
    def mobius_matvec_clamped(m, x, k=1.0, max_norm=150.0):
        """Safe mobius matrix-vector multiplication."""
        u = LorentzMathJAX.logmap0(x, k=k)
        mu = jnp.matmul(u, jnp.swapaxes(m, -1, -2))
        
        mu_abs = jnp.abs(mu)
        scale = jnp.where(mu_abs > max_norm, max_norm / (mu_abs + 1e-10), 1.0)
        mu_clamped = mu * scale
        
        return LorentzMathJAX.expmap0(mu_clamped, k=k)

    @staticmethod
    def mobius_scalar_mult(r, x, k=1.0, max_norm=150.0):
        """Safe mobius scalar multiplication."""
        u = LorentzMathJAX.logmap0(x, k=k)
        mu = u * r
        
        mu_abs = jnp.abs(mu)
        scale = jnp.where(mu_abs > max_norm, max_norm / (mu_abs + 1e-10), 1.0)
        mu_clamped = mu * scale
        
        return LorentzMathJAX.expmap0(mu_clamped, k=k)

    @staticmethod
    def apply_spatial_clamp(x, k=1.0, spatial_clamp=150.0, eps=1e-5):
        """Spatial clamping ensuring manifold constraint adherence."""
        max_norm = (spatial_clamp - eps) / (k**0.5)
        x_spatial = x[..., 1:]
        spatial_norm = jnp.linalg.norm(x_spatial, axis=-1, keepdims=True)
        
        scale = jnp.where(
            spatial_norm > max_norm,
            (max_norm / (spatial_norm + 1e-10)),
            1.0
        )
        x_spatial_scaled = x_spatial * scale
        x_0 = jnp.sqrt(k + jnp.sum(x_spatial_scaled**2, axis=-1, keepdims=True))
        return jnp.concatenate([x_0, x_spatial_scaled], axis=-1)

    @staticmethod
    def check_manifold_violation(x, k=1.0, name=""):
        """Check deviations from the hyperboloid constraint B(x, x) = -k."""
        x0 = x[..., 0:1]
        spatial_sq = jnp.sum(x[..., 1:]**2, axis=-1, keepdims=True)
        violation = jnp.abs(-x0**2 + spatial_sq + k)
        max_v = jnp.max(jnp.real(violation))
        has_nan = jnp.isnan(violation).any()
        
        def print_nan(_):
            jax.debug.print("CRITICAL: NaN detected at {name}", name=name)
            
        def print_warn(_):
            jax.debug.print("Warning: High manifold violation at {name}: {max_v:.6f}", name=name, max_v=max_v)
            
        def do_nothing(_):
            pass

        jax.lax.cond(has_nan, print_nan, do_nothing, operand=None)
        is_violating = jnp.logical_and(max_v > 1e-2, jnp.logical_not(has_nan))
        jax.lax.cond(is_violating, print_warn, do_nothing, operand=None)
        return has_nan
