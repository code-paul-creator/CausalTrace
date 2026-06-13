"""
Visualization utilities for CausalTrace results.

All plots return matplotlib Figure objects so they can be embedded
in the Streamlit UI or saved to disk.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


def plot_patching_heatmap(
    attn_recovery: np.ndarray,
    mlp_recovery: np.ndarray,
    title: str = "Activation Patching — Causal Recovery by Layer",
    figsize: Tuple[int, int] = (12, 4),
):
    """
    Side-by-side bar charts showing causal recovery for attention and MLP
    at each layer.

    Parameters
    ----------
    attn_recovery : np.ndarray, shape (n_layers,)
    mlp_recovery  : np.ndarray, shape (n_layers,)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for ax, data, label, cmap in zip(
        axes,
        [attn_recovery, mlp_recovery],
        ["Attention Output", "MLP Output"],
        ["Blues", "Oranges"],
    ):
        colors = cm.get_cmap(cmap)(data)
        bars = ax.barh(range(len(data)), data, color=colors)
        ax.set_xlabel("Recovery Score (0 = no effect, 1 = full recovery)")
        ax.set_ylabel("Layer")
        ax.set_title(label)
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        # Annotate top-3 layers
        top3 = np.argsort(data)[-3:]
        for idx in top3:
            ax.text(
                data[idx] + 0.01,
                idx,
                f"{data[idx]:.2f}",
                va="center",
                fontsize=8,
                color="black",
            )

    plt.tight_layout()
    return fig


def plot_logit_lens(
    per_layer_top_tokens: List[dict],
    target_token: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
):
    """
    Heatmap of top-1 token probability at each layer.
    Highlights `target_token` if provided.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    layers = [e["layer"] for e in per_layer_top_tokens]
    top_tokens = [e["top_tokens"][0][0].strip() for e in per_layer_top_tokens]
    top_probs = [e["top_tokens"][0][1] for e in per_layer_top_tokens]

    fig, ax = plt.subplots(figsize=figsize)

    colors = []
    for tok, prob in zip(top_tokens, top_probs):
        if target_token and tok.strip() == target_token.strip():
            colors.append((0.2, 0.6, 0.3, prob + 0.3))  # green when correct
        else:
            colors.append((0.2, 0.4, 0.8, prob + 0.1))  # blue otherwise

    ax.barh(layers, top_probs, color=colors)
    ax.set_yticks(layers)
    ax.set_yticklabels([f"L{l}: '{t}'" for l, t in zip(layers, top_tokens)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Probability of top token")
    ax.set_title("Logit Lens: Per-Layer Top Prediction", fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    return fig


def plot_token_trajectory(
    probs: np.ndarray,
    target_token: str,
    prompt: str,
    figsize: Tuple[int, int] = (10, 4),
):
    """
    Line plot of a specific token's probability across layers.
    Reveals exactly when the model "commits" to that token.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    layers = np.arange(len(probs))

    ax.plot(layers, probs, marker="o", markersize=4, linewidth=1.5, color="#2563eb")
    ax.fill_between(layers, probs, alpha=0.1, color="#2563eb")

    peak_layer = np.argmax(probs)
    ax.axvline(peak_layer, color="#dc2626", linestyle="--", linewidth=1, label=f"Peak at L{peak_layer}")
    ax.scatter([peak_layer], [probs[peak_layer]], color="#dc2626", zorder=5)

    ax.set_xlabel("Layer")
    ax.set_ylabel(f"P('{target_token.strip()}')")
    ax.set_title(f"Token Trajectory — '{target_token.strip()}'\n\"{prompt}\"", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig


def plot_scrub_comparison(
    scrub_results: List,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Bar chart comparing multiple scrubbing experiments side by side.
    Shows original, scrubbed, and random baseline for each circuit hypothesis.
    """
    import matplotlib.pyplot as plt

    n = len(scrub_results)
    labels = labels or [str(r.circuit) for r in scrub_results]
    x = np.arange(n)
    width = 0.25

    originals = [r.original_metric for r in scrub_results]
    scrubbed = [r.scrubbed_metric for r in scrub_results]
    baselines = [r.random_baseline for r in scrub_results]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, originals, width, label="Original", color="#2563eb", alpha=0.85)
    ax.bar(x, scrubbed, width, label="Scrubbed (circuit only)", color="#16a34a", alpha=0.85)
    ax.bar(x + width, baselines, width, label="Random baseline", color="#9ca3af", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Metric score")
    ax.set_title("Causal Scrubbing — Circuit Sufficiency Test", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return fig
