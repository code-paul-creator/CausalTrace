"""
CausalTrace — Example: Tracing Factual Recall in GPT-2

This script replicates a simplified version of the ROME paper's
causal tracing experiment on GPT-2.

Question answered: "Which layers store the fact that the Eiffel Tower is in Paris?"
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import matplotlib.pyplot as plt

from causaltrace import ModelWrapper, ActivationPatcher, LogitLens

# ── 1. Load model ──────────────────────────────────────────────────────────────

print("Loading GPT-2...")
model = ModelWrapper.from_pretrained("gpt2", device="cpu")
print(f"  Layers: {model.n_layers}, d_model: {model.d_model}")

# ── 2. Logit Lens: see how prediction forms layer by layer ─────────────────────

print("\n--- Logit Lens ---")
lens = LogitLens(model)
result = lens.run("The Eiffel Tower is located in the city of", top_k=3)
print(result)

# ── 3. Token Trajectory: track P(" Paris") across layers ──────────────────────

print("\n--- Token Trajectory ---")
probs = lens.token_trajectory(
    "The Eiffel Tower is located in the city of",
    target_token=" Paris",
)
import numpy as np
peak = np.argmax(probs)
print(f"P(' Paris') peaks at layer {peak} with value {probs[peak]:.4f}")

from causaltrace.viz.plots import plot_token_trajectory
fig = plot_token_trajectory(probs, " Paris", "The Eiffel Tower is located in the city of")
fig.savefig("token_trajectory.png", dpi=150, bbox_inches="tight")
print("Saved: token_trajectory.png")

# ── 4. Activation Patching: causal intervention ────────────────────────────────

print("\n--- Activation Patching ---")
patcher = ActivationPatcher(model)
result = patcher.patch_sweep(
    prompt_clean="The Eiffel Tower is located in the city of",
    prompt_corrupted="xjqz Eiffel Tower is located in the city of",
    target_token=" Paris",
    show_progress=True,
)

print(f"\nClean logit:     {result.clean_logit:.4f}")
print(f"Corrupted logit: {result.corrupted_logit:.4f}")
print(f"\nTop-3 attention layers by causal recovery:")
top3_attn = np.argsort(result.attn_recovery)[-3:][::-1]
for l in top3_attn:
    print(f"  Layer {l:2d}: {result.attn_recovery[l]:.4f}")

print(f"\nTop-3 MLP layers by causal recovery:")
top3_mlp = np.argsort(result.mlp_recovery)[-3:][::-1]
for l in top3_mlp:
    print(f"  Layer {l:2d}: {result.mlp_recovery[l]:.4f}")

from causaltrace.viz.plots import plot_patching_heatmap
fig2 = plot_patching_heatmap(result.attn_recovery, result.mlp_recovery)
fig2.savefig("patching_heatmap.png", dpi=150, bbox_inches="tight")
print("\nSaved: patching_heatmap.png")

print("\nDone.")
