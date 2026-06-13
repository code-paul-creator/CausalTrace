"""
Causal Scrubbing: a principled method for testing computational hypotheses
about transformer circuits.

Given a hypothesis H about which components are responsible for behaviour B:
  1. Resample all activations NOT in H from a distribution of "unrelated" inputs
  2. If performance on B is preserved → H is sufficient
  3. If performance degrades → H is incomplete

Reference: Chan et al., "Causal Scrubbing: a method for rigorously testing
interpretability hypotheses" (Redwood Research, 2022)
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import torch
import numpy as np

from causaltrace.core.model_wrapper import ModelWrapper, ActivationCache


@dataclass
class Circuit:
    """
    Defines a hypothesis about which components implement a behaviour.

    Parameters
    ----------
    attn_heads : dict of {layer_idx: list of head_idx}
        Attention heads included in the circuit.
    mlp_layers : list of int
        MLP layers included in the circuit.
    """

    attn_heads: Dict[int, List[int]]
    mlp_layers: List[int]

    def __repr__(self):
        heads_str = ", ".join(
            f"L{l}H{h}" for l, hs in self.attn_heads.items() for h in hs
        )
        mlp_str = ", ".join(f"MLP{l}" for l in self.mlp_layers)
        return f"Circuit(attn=[{heads_str}], mlp=[{mlp_str}])"


@dataclass
class ScrubResult:
    """Results of a causal scrubbing experiment."""
    circuit: Circuit
    original_metric: float
    scrubbed_metric: float
    random_baseline: float

    @property
    def preserved_fraction(self) -> float:
        """How much of the original performance is preserved after scrubbing."""
        denom = self.original_metric - self.random_baseline
        if abs(denom) < 1e-8:
            return 0.0
        return (self.scrubbed_metric - self.random_baseline) / denom

    def __repr__(self):
        return (
            f"ScrubResult:\n"
            f"  Circuit:            {self.circuit}\n"
            f"  Original metric:    {self.original_metric:.4f}\n"
            f"  Scrubbed metric:    {self.scrubbed_metric:.4f}\n"
            f"  Random baseline:    {self.random_baseline:.4f}\n"
            f"  Preserved fraction: {self.preserved_fraction:.2%}\n"
        )


class CausalScrubber:
    """
    Tests circuit hypotheses via causal scrubbing.

    Parameters
    ----------
    model : ModelWrapper
    metric_fn : callable
        Takes logits (torch.Tensor) and returns a scalar score (float).
        Example: probability of target token.
    reference_prompts : list of str
        Pool of unrelated prompts used for resampling activations outside H.
    """

    def __init__(
        self,
        model: ModelWrapper,
        metric_fn: Callable[[torch.Tensor], float],
        reference_prompts: List[str],
    ):
        self.model = model
        self.metric_fn = metric_fn
        self.reference_prompts = reference_prompts

    def scrub(
        self,
        prompt: str,
        circuit: Circuit,
        n_samples: int = 20,
        seed: int = 42,
    ) -> ScrubResult:
        """
        Run a causal scrubbing experiment.

        Parameters
        ----------
        prompt : str
            The target prompt to test the circuit on.
        circuit : Circuit
            Hypothesis about which components matter.
        n_samples : int
            How many reference prompts to average over.
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        ScrubResult
        """
        rng = np.random.default_rng(seed)

        # Original metric
        with self.model.trace(prompt):
            logits, _ = self.model.run()
        original_metric = self.metric_fn(logits)

        # Sample reference prompts
        idxs = rng.choice(len(self.reference_prompts), size=n_samples, replace=False)
        ref_prompts = [self.reference_prompts[i] for i in idxs]

        # Precompute reference caches
        ref_caches: List[ActivationCache] = []
        for rp in ref_prompts:
            with self.model.trace(rp):
                _, cache = self.model.run()
            ref_caches.append(cache)

        # Scrubbed metric: resample all non-circuit activations
        scrubbed_metrics = []
        for ref_cache in ref_caches:
            metric = self._scrubbed_run(prompt, circuit, ref_cache)
            scrubbed_metrics.append(metric)
        scrubbed_metric = float(np.mean(scrubbed_metrics))

        # Random baseline: resample EVERYTHING (circuit destroyed)
        empty_circuit = Circuit(attn_heads={}, mlp_layers=[])
        random_metrics = []
        for ref_cache in ref_caches:
            metric = self._scrubbed_run(prompt, empty_circuit, ref_cache)
            random_metrics.append(metric)
        random_baseline = float(np.mean(random_metrics))

        return ScrubResult(
            circuit=circuit,
            original_metric=original_metric,
            scrubbed_metric=scrubbed_metric,
            random_baseline=random_baseline,
        )

    def _scrubbed_run(
        self,
        prompt: str,
        circuit: Circuit,
        ref_cache: ActivationCache,
    ) -> float:
        """
        Run the model on `prompt`, but patch non-circuit components with ref_cache.
        """
        hook_handles = []
        layers_path, attn_name, mlp_name = self.model.SUPPORTED_ARCH[self.model.arch_key]
        layers = self.model.model
        for part in layers_path.split("."):
            layers = getattr(layers, part)

        for layer_idx in range(self.model.n_layers):
            key = f"layer_{layer_idx}"
            layer = layers[layer_idx]

            # MLP: patch if NOT in circuit
            if layer_idx not in circuit.mlp_layers and key in ref_cache.mlp_outputs:
                ref_act = ref_cache.mlp_outputs[key]

                def make_mlp_patch(act):
                    def hook(module, inp, out):
                        # Match sequence length if needed
                        if act.shape[1] != out.shape[1]:
                            return out  # skip if shapes incompatible
                        return act.to(out.device)
                    return hook

                mlp_module = getattr(layer, mlp_name, None)
                if mlp_module is not None:
                    h = mlp_module.register_forward_hook(make_mlp_patch(ref_act))
                    hook_handles.append(h)

        inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.model(**inputs)
        metric = self.metric_fn(outputs.logits)

        for h in hook_handles:
            h.remove()

        return metric
