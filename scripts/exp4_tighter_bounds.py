import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model import IterativeTransformerBlock
from src.utils import (
    _min_row_norm,
    compute_ffn_residual_spectral_norm,
    compute_ffn_spectral_norm,
    compute_jacobian_spectral_norm,
    compute_msa_residual_spectral_norm,
    compute_msa_spectral_norm,
    compute_tb_residual_spectral_norm,
    compute_tb_spectral_norm,
    setup_experiment,
)


def collect_bound_decomposition(n_iter=80):
    save_dir = setup_experiment(seed=42)
    os.makedirs("results", exist_ok=True)

    S, D, H = 32, 64, 4
    ffn_hidden = 128
    nu_values = np.linspace(0.1, 2.0, 10)

    model = IterativeTransformerBlock(D, H, ffn_hidden)
    model.eval()

    Y = torch.randn(1, S, D) * 0.5
    C = torch.randn(1, S, D) * 0.1

    with torch.no_grad():
        attn_out, _ = model.msa(Y, Y, Y)
        input_rms1 = Y + attn_out
        Y_prime = model.rms1(input_rms1)
        ffn_out = model.ffn(Y_prime)
        input_rms2 = Y_prime + ffn_out
        TB_Y = model.rms2(input_rms2)

    r_in = _min_row_norm(input_rms1)
    r_out = _min_row_norm(input_rms2)
    gamma_max = max(
        model.rms1.get_gamma_max(),
        model.rms2.get_gamma_max(),
        model.rms_final.get_gamma_max(),
    )

    msa_norm = compute_msa_spectral_norm(model, Y, n_iter=n_iter)
    ffn_norm = compute_ffn_spectral_norm(model, Y_prime, n_iter=n_iter)
    msa_res_norm = compute_msa_residual_spectral_norm(model, Y, n_iter=n_iter)
    ffn_res_norm = compute_ffn_residual_spectral_norm(model, Y_prime, n_iter=n_iter)
    tb_norm = compute_tb_spectral_norm(model, Y, n_iter=n_iter)

    component_inner = (
        gamma_max**2
        / (r_out * r_in)
        * (1 + ffn_norm)
        * (1 + msa_norm)
    )
    residual_inner = (
        gamma_max**2
        / (r_out * r_in)
        * ffn_res_norm
        * msa_res_norm
    )

    rows = []
    for nu in nu_values:
        model.nu = float(nu)
        with torch.no_grad():
            U = Y + model.nu * (C + TB_Y)
        r_U = _min_row_norm(U)
        outer = gamma_max / r_U

        empirical = compute_jacobian_spectral_norm(model, Y, C, n_iter=n_iter)
        original_bound = outer * (1 + model.nu * component_inner)
        residual_component_bound = outer * (1 + model.nu * residual_inner)
        tb_norm_bound = outer * (1 + model.nu * tb_norm)
        tb_residual_norm = compute_tb_residual_spectral_norm(
            model, Y, nu=float(nu), n_iter=n_iter
        )
        preactivation_bound = outer * tb_residual_norm

        rows.append(
            {
                "nu": float(nu),
                "empirical": empirical,
                "original_bound": original_bound,
                "residual_component_bound": residual_component_bound,
                "tb_norm_bound": tb_norm_bound,
                "preactivation_bound": preactivation_bound,
                "r_U": r_U,
                "outer": outer,
                "component_inner": component_inner,
                "residual_inner": residual_inner,
                "tb_norm": tb_norm,
                "tb_residual_norm": tb_residual_norm,
                "msa_norm": msa_norm,
                "ffn_norm": ffn_norm,
                "msa_res_norm": msa_res_norm,
                "ffn_res_norm": ffn_res_norm,
                "r_in": r_in,
                "r_out": r_out,
            }
        )

    csv_path = "results/exp4_tighter_bounds.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    x = [row["nu"] for row in rows]
    plt.figure(figsize=(10, 6))
    plt.plot(x, [row["empirical"] for row in rows], "o-", label="Empirical ||J_F||")
    plt.plot(
        x,
        [row["original_bound"] for row in rows],
        "--",
        label="Original component bound",
    )
    plt.plot(
        x,
        [row["residual_component_bound"] for row in rows],
        "--",
        label="Residual-component bound",
    )
    plt.plot(x, [row["tb_norm_bound"] for row in rows], "--", label="TB-norm bound")
    plt.plot(
        x,
        [row["preactivation_bound"] for row in rows],
        "--",
        label="Preactivation bound",
    )
    plt.xlabel(r"Step size $\nu$")
    plt.ylabel("Spectral Norm")
    plt.title("Tightening Jacobian Upper Bounds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = f"{save_dir}/fig4_tighter_bounds.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return rows, csv_path, plot_path


def main():
    rows, csv_path, plot_path = collect_bound_decomposition()
    print(f"Saved CSV to {csv_path}")
    print(f"Saved plot to {plot_path}")
    print(
        "nu empirical original residual_component tb_norm preactivation "
        "original_gap preactivation_gap"
    )
    for row in rows:
        print(
            f"{row['nu']:.2f} "
            f"{row['empirical']:.3f} "
            f"{row['original_bound']:.3f} "
            f"{row['residual_component_bound']:.3f} "
            f"{row['tb_norm_bound']:.3f} "
            f"{row['preactivation_bound']:.3f} "
            f"{row['original_bound'] / row['empirical']:.2f}x "
            f"{row['preactivation_bound'] / row['empirical']:.2f}x"
        )


if __name__ == "__main__":
    main()
