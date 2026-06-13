"""
Activation Patching: causal intervention to identify which components
are responsible for a factual output.

Algorithm
---------
1. Run the model on a "clean" prompt  → cache clean activations
2. Run the model on a "corrupted" prompt → cache corrupted activations
3. For each component (layer L, head H or MLP):
   a. Start from the corrupted run
   b. Patch in the clean activation at component (L, H)
   c. Measure how much the target token's logit recovers
4. Components with high recovery = causally important

Reference: Meng et al., "Locating and Editing Factual Associations in GPT" (NeurIPS 2022)
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

from causaltrace.core.model_wrapper import ModelWrapper, ActivationCache


@dataclass
class PatchingResult:
    """Holds the outcome of a full patching sweep."""
    # Shape: (n_layers,) for MLP; (n_layers, n_heads) for attention
    attn_recovery: np.ndarray
    mlp_recovery: np.ndarray
    residual_recovery: np.ndarray

    clean_logit: float
    corrupted_logit: float
    target_token: str
    prompt_clean: str
    prompt_corrupted: str


class ActivationPatcher:
    """
    Performs causal tracing via activation patching over all layers.

    Parameters
    ----------
    model : ModelWrapper
        Wrapped model with registered hooks.
    """

    def __init__(self, model: ModelWrapper):
        self.model = model

    def patch_sweep(
        self,
        prompt_clean: str,
        prompt_corrupted: str,
        target_token: str,
        patch_types: List[str] = ("attn", "mlp", "residual"),
        show_progress: bool = True,
    ) -> PatchingResult:
        """
        Full patching sweep across all layers and heads.

        Parameters
        ----------
        prompt_clean : str
            The factually correct prompt (e.g., "The Eiffel Tower is in Paris").
        prompt_corrupted : str
            Noisy or counterfactual variant (e.g., "xjqz Eiffel Tower is in Paris").
        target_token : str
            The token whose logit we track (e.g., " Paris").
        patch_types : list of str
            Which components to patch: 'attn', 'mlp', 'residual'.

        Returns
        -------
        PatchingResult
        """
        target_id = self.model.tokenizer.encode(target_token)[-1]

        # Step 1: Clean run
        with self.model.trace(prompt_clean):
            logits_clean, cache_clean = self.model.run()
        clean_logit = self._get_logit(logits_clean, target_id)

        # Step 2: Corrupted run
        with self.model.trace(prompt_corrupted):
            logits_corrupt, cache_corrupt = self.model.run()
        corrupt_logit = self._get_logit(logits_corrupt, target_id)

        n_layers = self.model.n_layers

        attn_recovery = np.zeros(n_layers)
        mlp_recovery = np.zeros(n_layers)
        residual_recovery = np.zeros(n_layers)

        layers = range(n_layers)
        if show_progress:
            layers = tqdm(layers, desc="Patching layers")

        for layer_idx in layers:
            key = f"layer_{layer_idx}"

            if "attn" in patch_types and key in cache_clean.attn_outputs:
                logit = self._patch_and_run(
                    prompt_corrupted,
                    component_key=key,
                    cache_source=cache_clean,
                    cache_type="attn_outputs",
                    target_id=target_id,
                )
                attn_recovery[layer_idx] = self._recovery_score(
                    corrupt_logit, clean_logit, logit
                )

            if "mlp" in patch_types and key in cache_clean.mlp_outputs:
                logit = self._patch_and_run(
                    prompt_corrupted,
                    component_key=key,
                    cache_source=cache_clean,
                    cache_type="mlp_outputs",
                    target_id=target_id,
                )
                mlp_recovery[layer_idx] = self._recovery_score(
                    corrupt_logit, clean_logit, logit
                )

            if "residual" in patch_types and key in cache_clean.residual_stream:
                logit = self._patch_and_run(
                    prompt_corrupted,
                    component_key=key,
                    cache_source=cache_clean,
                    cache_type="residual_stream",
                    target_id=target_id,
                )
                residual_recovery[layer_idx] = self._recovery_score(
                    corrupt_logit, clean_logit, logit
                )

        return PatchingResult(
            attn_recovery=attn_recovery,
            mlp_recovery=mlp_recovery,
            residual_recovery=residual_recovery,
            clean_logit=clean_logit,
            corrupted_logit=corrupt_logit,
            target_token=target_token,
            prompt_clean=prompt_clean,
            prompt_corrupted=prompt_corrupted,
        )

    def _patch_and_run(
        self,
        prompt_corrupted: str,
        component_key: str,
        cache_source: ActivationCache,
        cache_type: str,
        target_id: int,
    ) -> float:
        """Patch a single component and return the target token logit."""
        clean_act = getattr(cache_source, cache_type)[component_key]
        hook_handles = []

        def patch_hook(module, inp, out):
            if isinstance(out, tuple):
                patched = (clean_act.to(out[0].device),) + out[1:]
                return patched
            return clean_act.to(out.device)

        # Attach a single-shot patch hook
        layers_path = self.model.SUPPORTED_ARCH[self.model.arch_key][0]
        layers = self.model.model
        for part in layers_path.split("."):
            layers = getattr(layers, part)

        layer_idx = int(component_key.split("_")[1])
        layer = layers[layer_idx]

        if cache_type == "residual_stream":
            h = layer.register_forward_hook(patch_hook)
        elif cache_type == "attn_outputs":
            attn_name = self.model.SUPPORTED_ARCH[self.model.arch_key][1]
            h = getattr(layer, attn_name).register_forward_hook(patch_hook)
        else:  # mlp_outputs
            mlp_name = self.model.SUPPORTED_ARCH[self.model.arch_key][2]
            h = getattr(layer, mlp_name).register_forward_hook(patch_hook)

        hook_handles.append(h)

        inputs = self.model.tokenizer(prompt_corrupted, return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            outputs = self.model.model(**inputs)
        logit = self._get_logit(outputs.logits, target_id)

        for h in hook_handles:
            h.remove()

        return logit

    @staticmethod
    def _get_logit(logits: torch.Tensor, token_id: int) -> float:
        return logits[0, -1, token_id].item()

    @staticmethod
    def _recovery_score(corrupt: float, clean: float, patched: float) -> float:
        """
        Normalized recovery score in [0, 1].
        0 = no recovery (patch had no effect), 1 = full recovery (patch fully restored clean output).
        """
        denom = clean - corrupt
        if abs(denom) < 1e-8:
            return 0.0
        return (patched - corrupt) / denom
