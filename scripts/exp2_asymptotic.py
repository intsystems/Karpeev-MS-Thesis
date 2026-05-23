import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.model import IterativeTransformerBlock
from src.utils import setup_experiment, compute_jacobian_spectral_norm


def main():
    save_dir = setup_experiment(seed=42)
    print("Running Experiment 2: Asymptotic Behavior (O(1))...")

    S, D, H = 16, 32, 4
    ffn_hidden = 64
    nu_values = np.logspace(-1, 2, 20)
    n_seeds = 10

    all_norms = np.full((n_seeds, len(nu_values)), np.nan)

    for s in tqdm(range(n_seeds), desc="Seeds"):
        torch.manual_seed(1000 + s)
        model = IterativeTransformerBlock(D, H, ffn_hidden)
        model.eval()
        Y = torch.randn(1, S, D) * 0.5
        C = torch.randn(1, S, D) * 0.1

        for j, nu in enumerate(nu_values):
            model.nu = float(nu)
            all_norms[s, j] = compute_jacobian_spectral_norm(model, Y, C)

    norms = np.nanmean(all_norms, axis=0)
    stds = np.nanstd(all_norms, axis=0)

    plt.figure()
    plt.plot(nu_values, norms, "o-", color="#1f77b4")
    plt.fill_between(nu_values, norms - stds, norms + stds, alpha=0.3, color="#1f77b4")
    plt.xscale("log")
    plt.xlabel(r"Step size $\nu$ (log scale)")
    plt.ylabel(r"Spectral Norm $\|J_F\|_2$")
    plt.title(r"Asymptotic Stability ($O(1)$) as $\nu \to \infty$")

    tail = norms[-5:]
    if not np.all(np.isnan(tail)):
        plt.axhline(
            y=np.nanmean(tail),
            color="r",
            linestyle="--",
            label="Asymptote",
            alpha=0.5,
        )
        plt.legend()

    plt.grid(True, which="both", ls="-")
    plt.tight_layout()

    save_path = f"{save_dir}/fig2_asymptotic.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved Figure 2 to {save_path}")


if __name__ == "__main__":
    main()
