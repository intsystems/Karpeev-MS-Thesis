"""Validation of Theorem 3.1 / 3.2 with pretrained weights.

We instantiate our IterativeTransformerBlock with MSA/FFN weights copied from a
middle layer of pretrained DeiT-tiny, and drive it with patch+position embeddings
of a real CIFAR-10 image (resized to 224 and ImageNet-normalized, as DeiT expects).
The normalization stays RMSNorm (Post-Norm) — exactly the setting of the theorem.
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.pretrained import (
    build_block_from_deit,
    cifar10_to_deit_input,
    patch_embeddings_from_image,
)
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

import torchvision
import torchvision.transforms as T


def get_cifar_image(seed: int = 0):
    """Return one CIFAR-10 image as a tensor (3, 32, 32) in [-1, 1]."""
    tfm = T.Compose(
        [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    ds = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=tfm
    )
    g = torch.Generator().manual_seed(seed)
    idx = int(torch.randint(0, len(ds), (1,), generator=g).item())
    img, _ = ds[idx]
    return img


def precompute_bound_terms(block, Y, n_iter):
    """Collect all bound factors that do not depend on the context C or step size nu."""
    with torch.no_grad():
        attn_out, _ = block.msa(Y, Y, Y)
        input_rms1 = Y + attn_out
        Y_prime = block.rms1(input_rms1)

        ffn_out = block.ffn(Y_prime)
        input_rms2 = Y_prime + ffn_out
        TB_Y = block.rms2(input_rms2)

    r_in = _min_row_norm(input_rms1)
    r_out = _min_row_norm(input_rms2)

    msa_norm = compute_msa_spectral_norm(block, Y, n_iter=n_iter)
    ffn_norm = compute_ffn_spectral_norm(block, Y_prime, n_iter=n_iter)
    msa_res_norm = compute_msa_residual_spectral_norm(block, Y, n_iter=n_iter)
    ffn_res_norm = compute_ffn_residual_spectral_norm(block, Y_prime, n_iter=n_iter)
    tb_norm = compute_tb_spectral_norm(block, Y, n_iter=n_iter)

    gamma_max = max(
        block.rms1.get_gamma_max(),
        block.rms2.get_gamma_max(),
        block.rms_final.get_gamma_max(),
    )

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

    return {
        "TB_Y": TB_Y,
        "r_in": r_in,
        "r_out": r_out,
        "gamma_max": gamma_max,
        "msa_norm": msa_norm,
        "ffn_norm": ffn_norm,
        "msa_res_norm": msa_res_norm,
        "ffn_res_norm": ffn_res_norm,
        "tb_norm": tb_norm,
        "component_inner": component_inner,
        "residual_inner": residual_inner,
    }


def compute_bound_row(block, Y, C, terms, nu, context_name, n_iter):
    block.nu = float(nu)
    with torch.no_grad():
        U = Y + block.nu * (C + terms["TB_Y"])

    r_U = _min_row_norm(U)
    outer = terms["gamma_max"] / r_U
    empirical = compute_jacobian_spectral_norm(block, Y, C, n_iter=n_iter)

    original_bound = outer * (1 + block.nu * terms["component_inner"])
    residual_component_bound = outer * (1 + block.nu * terms["residual_inner"])
    tb_norm_bound = outer * (1 + block.nu * terms["tb_norm"])
    tb_residual_norm = compute_tb_residual_spectral_norm(
        block, Y, nu=float(nu), n_iter=n_iter
    )
    preactivation_bound = outer * tb_residual_norm

    return {
        "sweep": "bound",
        "context": context_name,
        "nu": float(nu),
        "empirical": empirical,
        "original_bound": original_bound,
        "residual_component_bound": residual_component_bound,
        "tb_norm_bound": tb_norm_bound,
        "preactivation_bound": preactivation_bound,
        "original_ratio": original_bound / empirical,
        "residual_component_ratio": residual_component_bound / empirical,
        "tb_norm_ratio": tb_norm_bound / empirical,
        "preactivation_ratio": preactivation_bound / empirical,
        "r_U": r_U,
        "r_in": terms["r_in"],
        "r_out": terms["r_out"],
        "msa_norm": terms["msa_norm"],
        "ffn_norm": terms["ffn_norm"],
        "msa_res_norm": terms["msa_res_norm"],
        "ffn_res_norm": terms["ffn_res_norm"],
        "tb_norm": terms["tb_norm"],
        "tb_residual_norm": tb_residual_norm,
    }


def compute_asymptotic_row(block, Y, C, nu, context_name, n_iter):
    block.nu = float(nu)
    empirical = compute_jacobian_spectral_norm(block, Y, C, n_iter=n_iter)
    return {
        "sweep": "asymptotic",
        "context": context_name,
        "nu": float(nu),
        "empirical": empirical,
        "original_bound": "",
        "residual_component_bound": "",
        "tb_norm_bound": "",
        "preactivation_bound": "",
        "original_ratio": "",
        "residual_component_ratio": "",
        "tb_norm_ratio": "",
        "preactivation_ratio": "",
        "r_U": "",
        "r_in": "",
        "r_out": "",
        "msa_norm": "",
        "ffn_norm": "",
        "msa_res_norm": "",
        "ffn_res_norm": "",
        "tb_norm": "",
        "tb_residual_norm": "",
    }


def plot_context_row(axes, context_name, bound_rows, asym_rows):
    ax1, ax2 = axes
    nu_bound = [row["nu"] for row in bound_rows]
    ax1.plot(
        nu_bound,
        [row["empirical"] for row in bound_rows],
        "o-",
        label=r"Empirical $\|J_F\|_2$",
        color="#1f77b4",
    )
    ax1.plot(
        nu_bound,
        [row["original_bound"] for row in bound_rows],
        "--",
        label="Original bound",
        color="#ff7f0e",
    )
    ax1.plot(
        nu_bound,
        [row["residual_component_bound"] for row in bound_rows],
        "--",
        label="Residual-component",
        color="#2ca02c",
    )
    ax1.plot(
        nu_bound,
        [row["tb_norm_bound"] for row in bound_rows],
        "--",
        label="TB-norm",
        color="#d62728",
    )
    ax1.plot(
        nu_bound,
        [row["preactivation_bound"] for row in bound_rows],
        "--",
        label="Preactivation",
        color="#9467bd",
    )
    ax1.set_xlabel(r"Step size $\nu$")
    ax1.set_ylabel("Spectral norm")
    ax1.set_title(f"Bound tightness ({context_name})")
    ax1.legend()
    ax1.grid(True)

    nu_asym = [row["nu"] for row in asym_rows]
    asym_norms = np.asarray([row["empirical"] for row in asym_rows])
    ax2.plot(nu_asym, asym_norms, "o-", color="#1f77b4")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Step size $\nu$ (log scale)")
    ax2.set_ylabel(r"$\|J_F\|_2$")
    ax2.set_title(f"Large-step behavior ({context_name})")
    tail = asym_norms[-5:]
    if np.all(np.isfinite(tail)):
        ax2.axhline(
            y=tail.mean(),
            color="r",
            linestyle="--",
            label="Tail mean",
            alpha=0.5,
        )
        ax2.legend()
    ax2.grid(True, which="both", ls="-")


def main():
    save_dir = setup_experiment(seed=42)
    os.makedirs("results", exist_ok=True)
    print("Running Experiment 5: Theorem validation with pretrained DeiT-tiny weights...")

    print("Loading DeiT-tiny and copying MSA/FFN weights into IterativeTransformerBlock...")
    block, deit = build_block_from_deit(layer_index=6)

    print("Producing patch embeddings Y from a CIFAR-10 image...")
    cifar_img = get_cifar_image(seed=0)
    deit_input = cifar10_to_deit_input(cifar_img)
    Y = patch_embeddings_from_image(deit, deit_input)
    Y = Y.detach()
    print(f"  Y shape: {tuple(Y.shape)} (B, S, D)")

    torch.manual_seed(0)
    contexts = {
        "random context": 0.1 * torch.randn_like(Y),
        "image-derived context": 0.1 * Y,
    }

    n_iter = 30
    print(f"Precomputing context-independent bound terms (power iterations={n_iter})...")
    terms = precompute_bound_terms(block, Y, n_iter=n_iter)

    # --- Bound validation (Theorem 3.1) ---
    nu_values_bound = np.linspace(0.1, 2.0, 10)
    nu_values_asym = np.logspace(-1, 2, 20)

    all_rows = []
    rows_by_context = {}
    for context_name, C in contexts.items():
        bound_rows = []
        asym_rows = []

        for nu in tqdm(nu_values_bound, desc=f"bound sweep ({context_name})"):
            row = compute_bound_row(block, Y, C, terms, nu, context_name, n_iter)
            bound_rows.append(row)
            all_rows.append(row)

        for nu in tqdm(nu_values_asym, desc=f"asymptotic sweep ({context_name})"):
            row = compute_asymptotic_row(block, Y, C, nu, context_name, n_iter)
            asym_rows.append(row)
            all_rows.append(row)

        rows_by_context[context_name] = (bound_rows, asym_rows)

    # --- Plot ---
    fig, axes = plt.subplots(len(contexts), 2, figsize=(15, 9), squeeze=False)
    for ax_row, (context_name, (bound_rows, asym_rows)) in zip(
        axes, rows_by_context.items()
    ):
        plot_context_row(ax_row, context_name, bound_rows, asym_rows)

    plt.tight_layout()
    save_path = f"{save_dir}/fig5_pretrained.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Figure 5 to {save_path}")

    csv_path = "results/exp5_pretrained.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved CSV to {csv_path}")

    print("\nbound vs empirical (pretrained DeiT-tiny linear weights):")
    print(
        f"  {'context':>21}  {'nu':>6}  {'empirical':>10}  "
        f"{'original':>10}  {'residual':>10}  {'preact':>10}"
    )
    for context_name, (bound_rows, _) in rows_by_context.items():
        for row in bound_rows:
            print(
                f"  {context_name:>21}  "
                f"{row['nu']:6.3f}  "
                f"{row['empirical']:10.4f}  "
                f"{row['original_bound']:10.4f}  "
                f"{row['residual_component_bound']:10.4f}  "
                f"{row['preactivation_bound']:10.4f}"
            )


if __name__ == "__main__":
    main()
