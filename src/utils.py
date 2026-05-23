import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd.functional import jvp as t_jvp, vjp as t_vjp
import torch


def setup_experiment(seed=42, save_dir="plots"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.figsize": (10, 6),
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "grid.alpha": 0.3,
        }
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def power_iteration_spectral_norm(forward_fn, x_template, n_iter=50, tol=1e-6):
    """Top singular value of the Jacobian of `forward_fn` via JVP/VJP power iteration.

    `forward_fn` maps a tensor of shape `x_template.shape` to a tensor of arbitrary
    shape; the Jacobian is materialized only through matrix-vector products.
    """
    x_flat = x_template.detach().reshape(-1).contiguous()
    in_shape = x_template.shape

    def f(y_flat):
        return forward_fn(y_flat.view(in_shape)).reshape(-1)

    v = torch.randn_like(x_flat)
    v = v / torch.linalg.vector_norm(v)
    sigma_prev = 0.0
    sigma = 0.0
    for _ in range(n_iter):
        try:
            _, Jv = t_jvp(f, (x_flat,), (v,))
            Jv_norm = torch.linalg.vector_norm(Jv)
            if Jv_norm.item() < 1e-12:
                return 0.0
            u = Jv / Jv_norm
            _, JTu = t_vjp(f, x_flat, u)
            JTu_norm = torch.linalg.vector_norm(JTu)
            if JTu_norm.item() < 1e-12:
                return Jv_norm.item()
            v = JTu / JTu_norm
            sigma = JTu_norm.item()
        except RuntimeError as e:
            warnings.warn(f"Power iteration failed, returning NaN. Details: {e}")
            return float("nan")
        if abs(sigma - sigma_prev) < tol * max(sigma, 1e-12):
            break
        sigma_prev = sigma
    return sigma


def compute_jacobian_spectral_norm(model, Y, C, n_iter=50):
    """Spectral norm of J_F via power iteration on the full step `F(Y) = RMS(Y + nu(C + TB(Y)))`."""

    def fwd(y):
        return model.forward_step(y, C)

    return power_iteration_spectral_norm(fwd, Y, n_iter=n_iter)


def compute_msa_spectral_norm(model, Y, n_iter=50):
    def fwd(y):
        out, _ = model.msa(y, y, y)
        return out

    return power_iteration_spectral_norm(fwd, Y, n_iter=n_iter)


def compute_ffn_spectral_norm(model, Y_prime, n_iter=50):
    def fwd(y):
        return model.ffn(y)

    return power_iteration_spectral_norm(fwd, Y_prime, n_iter=n_iter)


def compute_msa_residual_spectral_norm(model, Y, n_iter=50):
    def fwd(y):
        out, _ = model.msa(y, y, y)
        return y + out

    return power_iteration_spectral_norm(fwd, Y, n_iter=n_iter)


def compute_ffn_residual_spectral_norm(model, Y_prime, n_iter=50):
    def fwd(y):
        return y + model.ffn(y)

    return power_iteration_spectral_norm(fwd, Y_prime, n_iter=n_iter)


def compute_tb_spectral_norm(model, Y, n_iter=50):
    return power_iteration_spectral_norm(model.forward_subblocks, Y, n_iter=n_iter)


def compute_tb_residual_spectral_norm(model, Y, nu=None, n_iter=50):
    step = model.nu if nu is None else nu

    def fwd(y):
        return y + step * model.forward_subblocks(y)

    return power_iteration_spectral_norm(fwd, Y, n_iter=n_iter)


def compute_preactivation(model, Y, C):
    if C.dim() == 2:
        C = C.unsqueeze(0).expand(Y.shape[0], -1, -1)
    return Y + model.nu * (C + model.forward_subblocks(Y))


def compute_row_aware_preactivation_bound(model, Y, C, n_iter=50):
    """||D_U J_U|| where D_U keeps the final RMSNorm row-wise denominators.

    This tightens (gamma_max / min_i RMS(U_i)) * ||J_U|| by retaining each row's
    own RMS denominator instead of collapsing the outer RMSNorm to its worst row.
    """
    with torch.no_grad():
        U = compute_preactivation(model, Y, C)
        gamma_max = model.rms_final.get_gamma_max()
        row_rms = torch.sqrt(torch.mean(U**2, dim=-1, keepdim=True) + model.rms_final.eps)
        row_scale = gamma_max / row_rms

    def fwd(y):
        return row_scale * compute_preactivation(model, y, C)

    return power_iteration_spectral_norm(fwd, Y, n_iter=n_iter)


def compute_exact_outer_linearized_norm(model, Y, C, n_iter=50):
    """||J_RMS(U) J_U|| using the exact final RMSNorm Jacobian frozen at U.

    This is a sanity-check factorization of the empirical full-step Jacobian, not
    an independent upper bound.
    """
    with torch.no_grad():
        U = compute_preactivation(model, Y, C)
        gamma = model.rms_final.gamma.detach().view(*([1] * (U.dim() - 1)), -1)
        row_rms = torch.sqrt(torch.mean(U**2, dim=-1, keepdim=True) + model.rms_final.eps)

    def apply_final_rms_jacobian(v):
        radial = U * torch.mean(U * v, dim=-1, keepdim=True) / (row_rms**3)
        return gamma * (v / row_rms - radial)

    def fwd(y):
        return apply_final_rms_jacobian(compute_preactivation(model, y, C))

    return power_iteration_spectral_norm(fwd, Y, n_iter=n_iter)


def _min_row_norm(M):
    """min_i ||M[i,:]|| matched to the model's RMSNorm denominator.

    The model uses RMSNorm(u) = γ * u / sqrt(mean(u^2)) (src/model.py), so the
    per-row Jacobian bound is γ_max / RMS(row). For the theoretical bound to
    actually upper-bound the empirical ||J_F||_2 we must use the same row
    quantity that drives the RMSNorm denominator — i.e. min row RMS.
    """
    flat = M.reshape(-1, M.shape[-1])
    rms_per_row = torch.sqrt(torch.mean(flat ** 2, dim=-1))
    return rms_per_row.min().item()


def compute_theoretical_bound(model, Y, C):
    """Theorem 3.1: γ_max / r_U * (1 + ν * γ_max^2/(r_out r_in) * (1+F)(1+M)),
    with empirically measured M = ||J_MSA||, F = ||J_FFN|| and minimal row L2-norms.
    """
    with torch.no_grad():
        attn_out, _ = model.msa(Y, Y, Y)
        input_rms1 = Y + attn_out
        Y_prime = model.rms1(input_rms1)

        ffn_out = model.ffn(Y_prime)
        input_rms2 = Y_prime + ffn_out
        TB_Y = model.rms2(input_rms2)

        U = Y + model.nu * (C + TB_Y)

    r_in = _min_row_norm(input_rms1)
    r_out = _min_row_norm(input_rms2)
    r_U = _min_row_norm(U)

    M_msa = compute_msa_spectral_norm(model, Y)
    F_ffn = compute_ffn_spectral_norm(model, Y_prime)

    gamma_max = max(
        model.rms1.get_gamma_max(),
        model.rms2.get_gamma_max(),
        model.rms_final.get_gamma_max(),
    )

    bound = (gamma_max / r_U) * (
        1
        + model.nu
        * (gamma_max ** 2 / (r_out * r_in))
        * (1 + F_ffn)
        * (1 + M_msa)
    )
    return bound
