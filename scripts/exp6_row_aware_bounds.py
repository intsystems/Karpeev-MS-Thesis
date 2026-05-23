"""Row-aware refinement of the outer RMSNorm Jacobian bound.

The standard pre-RMSNorm bound uses a single worst-row denominator:

    ||J_F|| <= gamma_max / min_i RMS(U_i) * ||J_U||.

This experiment evaluates a tighter certificate that keeps the row-wise
denominators of the final RMSNorm:

    ||J_F|| <= ||D_U J_U||,
    D_U = blockdiag_i((gamma_max / RMS(U_i)) I_D).

The second bound is always at least as tight as the scalar pre-RMSNorm bound,
because ||D_U|| = gamma_max / min_i RMS(U_i).
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from src.model import IterativeTransformerBlock
from src.pretrained import (
    build_block_from_deit,
    cifar10_to_deit_input,
    patch_embeddings_from_image,
)
from src.utils import (
    compute_exact_outer_linearized_norm,
    compute_jacobian_spectral_norm,
    compute_preactivation,
    compute_row_aware_preactivation_bound,
    compute_tb_residual_spectral_norm,
    setup_experiment,
)


def get_cifar_image(seed=0):
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


def scalar_preactivation_bound(model, Y, C, n_iter):
    with torch.no_grad():
        U = compute_preactivation(model, Y, C)
        row_rms = torch.sqrt(torch.mean(U**2, dim=-1) + model.rms_final.eps)
        outer = model.rms_final.get_gamma_max() / row_rms.min().item()
    preactivation_norm = compute_tb_residual_spectral_norm(
        model, Y, nu=model.nu, n_iter=n_iter
    )
    return outer * preactivation_norm


def collect_rows(case_name, model, Y, C, nu_values, n_iter):
    rows = []
    for nu in tqdm(nu_values, desc=case_name):
        model.nu = float(nu)
        empirical = compute_jacobian_spectral_norm(model, Y, C, n_iter=n_iter)
        scalar_bound = scalar_preactivation_bound(model, Y, C, n_iter=n_iter)
        row_aware_bound = compute_row_aware_preactivation_bound(
            model, Y, C, n_iter=n_iter
        )
        exact_factorization = compute_exact_outer_linearized_norm(
            model, Y, C, n_iter=n_iter
        )
        rows.append(
            {
                "case": case_name,
                "nu": float(nu),
                "empirical": empirical,
                "scalar_preactivation_bound": scalar_bound,
                "row_aware_bound": row_aware_bound,
                "exact_factorization": exact_factorization,
                "scalar_ratio": scalar_bound / empirical,
                "row_aware_ratio": row_aware_bound / empirical,
                "exact_factorization_ratio": exact_factorization / empirical,
            }
        )
    return rows


def build_synthetic_case():
    torch.manual_seed(42)
    S, D, H = 32, 64, 4
    model = IterativeTransformerBlock(D, H, 128)
    model.eval()
    Y = torch.randn(1, S, D) * 0.5
    C = torch.randn(1, S, D) * 0.1
    return [("synthetic random context", model, Y, C)]


def build_pretrained_cases():
    block, deit = build_block_from_deit(layer_index=6)
    cifar_img = get_cifar_image(seed=0)
    deit_input = cifar10_to_deit_input(cifar_img)
    Y = patch_embeddings_from_image(deit, deit_input).detach()

    torch.manual_seed(0)
    random_context = 0.1 * torch.randn_like(Y)
    image_context = 0.1 * Y

    return [
        ("pretrained random context", block, Y, random_context),
        ("pretrained image-derived context", block, Y, image_context),
    ]


def plot_results(rows, plot_path):
    cases = list(dict.fromkeys(row["case"] for row in rows))
    fig, axes = plt.subplots(len(cases), 1, figsize=(10, 4 * len(cases)), squeeze=False)

    for ax, case in zip(axes[:, 0], cases):
        case_rows = [row for row in rows if row["case"] == case]
        x = [row["nu"] for row in case_rows]
        ax.plot(x, [row["empirical"] for row in case_rows], "o-", label="Empirical")
        ax.plot(
            x,
            [row["scalar_preactivation_bound"] for row in case_rows],
            "--",
            label="Scalar pre-RMS bound",
        )
        ax.plot(
            x,
            [row["row_aware_bound"] for row in case_rows],
            "--",
            label="Row-aware pre-RMS bound",
        )
        ax.plot(
            x,
            [row["exact_factorization"] for row in case_rows],
            ":",
            label="Exact outer factorization",
        )
        ax.set_title(case)
        ax.set_xlabel(r"Step size $\nu$")
        ax.set_ylabel("Spectral norm")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def main():
    save_dir = setup_experiment(seed=42)
    os.makedirs("results", exist_ok=True)

    nu_values = np.linspace(0.1, 2.0, 10)
    n_iter = 120

    rows = []
    for case_name, model, Y, C in build_synthetic_case():
        rows.extend(collect_rows(case_name, model, Y, C, nu_values, n_iter))

    for case_name, model, Y, C in build_pretrained_cases():
        rows.extend(collect_rows(case_name, model, Y, C, nu_values, n_iter))

    csv_path = "results/exp6_row_aware_bounds.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = f"{save_dir}/fig6_row_aware_bounds.png"
    plot_results(rows, plot_path)

    print(f"Saved CSV to {csv_path}")
    print(f"Saved plot to {plot_path}")
    print(
        f"{'case':>34} {'nu':>5} {'emp':>8} {'scalar':>8} "
        f"{'row':>8} {'exact':>8} {'row_gap':>8}"
    )
    for row in rows:
        print(
            f"{row['case']:>34} "
            f"{row['nu']:5.2f} "
            f"{row['empirical']:8.3f} "
            f"{row['scalar_preactivation_bound']:8.3f} "
            f"{row['row_aware_bound']:8.3f} "
            f"{row['exact_factorization']:8.3f} "
            f"{row['row_aware_ratio']:8.3f}x"
        )


if __name__ == "__main__":
    main()
