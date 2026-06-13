"""
CausalTrace Web UI — built with Streamlit. 

Run:
    streamlit run web/app.py
"""

import streamlit as st
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from causaltrace.core.model_wrapper import ModelWrapper
from causaltrace.core.patching import ActivationPatcher
from causaltrace.core.logit_lens import LogitLens
from causaltrace.viz.plots import (
    plot_patching_heatmap,
    plot_logit_lens,
    plot_token_trajectory,
)

st.set_page_config(
    page_title="CausalTrace",
    page_icon="🔬",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("🔬 CausalTrace")
st.sidebar.markdown("Mechanistic Interpretability Toolkit")

model_name = st.sidebar.selectbox(
    "Model",
    ["gpt2", "gpt2-medium", "distilgpt2"],
    help="Select a HuggingFace model to analyse.",
)

tool = st.sidebar.radio(
    "Analysis tool",
    ["Activation Patching", "Logit Lens", "Token Trajectory"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**CausalTrace** traces which attention heads and MLP layers "
    "are causally responsible for factual outputs."
)
st.sidebar.markdown("[GitHub](https://github.com/yourusername/CausalTrace)")

# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model(name: str) -> ModelWrapper:
    return ModelWrapper.from_pretrained(name, device="cpu")

model = load_model(model_name)

# ── Main area ──────────────────────────────────────────────────────────────────

st.title("🔬 CausalTrace")
st.caption(f"Model: `{model_name}` · {model.n_layers} layers · d_model={model.d_model}")

if tool == "Activation Patching":
    st.header("Activation Patching")
    st.markdown(
        "Identifies **which layers** are causally responsible for a factual output by "
        "measuring how much the target token's probability recovers when a clean "
        "activation is patched into a corrupted run."
    )

    col1, col2 = st.columns(2)
    with col1:
        prompt_clean = st.text_area(
            "Clean prompt (factually correct)",
            "The Eiffel Tower is located in the city of",
            height=80,
        )
        target_token = st.text_input("Target token", " Paris")

    with col2:
        prompt_corrupted = st.text_area(
            "Corrupted prompt (noisy/counterfactual)",
            "xjqz Eiffel Tower is located in the city of",
            height=80,
        )

    if st.button("Run Activation Patching", type="primary"):
        with st.spinner("Running causal trace…"):
            patcher = ActivationPatcher(model)
            result = patcher.patch_sweep(
                prompt_clean,
                prompt_corrupted,
                target_token,
                show_progress=False,
            )

        st.success(
            f"Clean logit: **{result.clean_logit:.3f}** | "
            f"Corrupted logit: **{result.corrupted_logit:.3f}**"
        )

        fig = plot_patching_heatmap(result.attn_recovery, result.mlp_recovery)
        st.pyplot(fig)

        st.subheader("Recovery scores by layer")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Attention layers** (top 5 by recovery)")
            top5_attn = np.argsort(result.attn_recovery)[-5:][::-1]
            for l in top5_attn:
                st.metric(f"Layer {l}", f"{result.attn_recovery[l]:.3f}")
        with col_b:
            st.markdown("**MLP layers** (top 5 by recovery)")
            top5_mlp = np.argsort(result.mlp_recovery)[-5:][::-1]
            for l in top5_mlp:
                st.metric(f"Layer {l}", f"{result.mlp_recovery[l]:.3f}")

elif tool == "Logit Lens":
    st.header("Logit Lens")
    st.markdown(
        "Projects intermediate residual stream states into vocabulary space to show "
        "how the model's prediction **evolves layer by layer**."
    )

    prompt = st.text_area("Prompt", "The capital of France is", height=80)
    top_k = st.slider("Top-k tokens per layer", 1, 10, 5)

    if st.button("Run Logit Lens", type="primary"):
        with st.spinner("Running logit lens…"):
            lens = LogitLens(model)
            result = lens.run(prompt, top_k=top_k)

        fig = plot_logit_lens(result.per_layer)
        st.pyplot(fig)

        st.subheader("Per-layer predictions")
        for entry in result.per_layer:
            tokens_str = " | ".join(
                f"`{t.strip()}` ({p:.3f})" for t, p in entry["top_tokens"]
            )
            st.markdown(f"**Layer {entry['layer']}** → {tokens_str}")

        st.markdown(f"**Final output** → `{result.final_top_tokens[0][0].strip()}` ({result.final_top_tokens[0][1]:.3f})")

elif tool == "Token Trajectory":
    st.header("Token Trajectory")
    st.markdown(
        "Plots the probability of a specific target token **across layers**, "
        "revealing the exact layer where the model commits to its answer."
    )

    prompt = st.text_area("Prompt", "The inventor of the telephone was", height=80)
    target_token = st.text_input("Target token to track", " Bell")

    if st.button("Plot Trajectory", type="primary"):
        with st.spinner("Computing trajectory…"):
            lens = LogitLens(model)
            probs = lens.token_trajectory(prompt, target_token)

        fig = plot_token_trajectory(probs, target_token, prompt)
        st.pyplot(fig)

        peak_layer = int(np.argmax(probs))
        st.info(
            f"**Peak probability** of `{target_token.strip()}` = "
            f"**{probs[peak_layer]:.3f}** at layer **{peak_layer}** / {model.n_layers - 1}"
        )
