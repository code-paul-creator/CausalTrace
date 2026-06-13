"""
Logit Lens: project intermediate residual stream states into vocabulary space
to see how the model's "prediction" evolves layer by layer.

Reference: nostalgebraist, "interpreting GPT: the logit lens" (2020)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

from causaltrace.core.model_wrapper import ModelWrapper, ActivationCache


class LogitLens:
    """
    Projects the residual stream at each layer through the final LayerNorm
    and unembedding matrix to produce per-layer token probability distributions.

    This reveals the "trajectory" of a prediction — at which layer does the
    model first commit to the correct answer?

    Parameters
    ----------
    model : ModelWrapper
    """

    def __init__(self, model: ModelWrapper):
        self.model = model
        self._ln_f = self._get_final_ln()
        self._unembed = self._get_unembed()

    def _get_final_ln(self):
        """Retrieve the final layer norm from the model."""
        m = self.model.model
        for attr in ("transformer.ln_f", "gpt_neox.final_layer_norm", "model.norm"):
            try:
                obj = m
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise AttributeError("Could not locate final LayerNorm. Add support for this architecture.")

    def _get_unembed(self):
        """Retrieve the unembedding (lm_head) matrix."""
        for attr in ("lm_head", "embed_out"):
            if hasattr(self.model.model, attr):
                return getattr(self.model.model, attr)
        raise AttributeError("Could not locate unembedding matrix.")

    def run(
        self,
        prompt: str,
        token_position: int = -1,
        top_k: int = 5,
    ) -> "LogitLensResult":
        """
        Run logit lens on a prompt.

        Parameters
        ----------
        prompt : str
            Input text.
        token_position : int
            Which token position to analyse. Defaults to last (-1).
        top_k : int
            Number of top tokens to return per layer.

        Returns
        -------
        LogitLensResult
        """
        with self.model.trace(prompt):
            logits, cache = self.model.run()

        n_layers = self.model.n_layers
        per_layer: List[Dict] = []

        for layer_idx in range(n_layers):
            key = f"layer_{layer_idx}"
            if key not in cache.residual_stream:
                continue

            hidden = cache.residual_stream[key][:, token_position, :].unsqueeze(0)

            with torch.no_grad():
                normed = self._ln_f(hidden)
                layer_logits = self._unembed(normed)[0, 0, :]
                probs = torch.softmax(layer_logits, dim=-1)

            top = torch.topk(probs, top_k)
            tokens_probs = [
                (self.model.tokenizer.decode([idx.item()]), prob.item())
                for idx, prob in zip(top.indices, top.values)
            ]
            per_layer.append({"layer": layer_idx, "top_tokens": tokens_probs})

        # Final layer
        final_probs = torch.softmax(logits[0, token_position, :], dim=-1)
        final_top = torch.topk(final_probs, top_k)
        final_tokens = [
            (self.model.tokenizer.decode([idx.item()]), prob.item())
            for idx, prob in zip(final_top.indices, final_top.values)
        ]

        return LogitLensResult(
            prompt=prompt,
            per_layer=per_layer,
            final_top_tokens=final_tokens,
            token_position=token_position,
        )

    def token_trajectory(
        self,
        prompt: str,
        target_token: str,
        token_position: int = -1,
    ) -> np.ndarray:
        """
        Return the probability of a specific token at each layer.
        Useful for plotting how quickly the model "locks in" on the answer.
        """
        target_id = self.model.tokenizer.encode(target_token)[-1]
        with self.model.trace(prompt):
            logits, cache = self.model.run()

        probs = []
        for layer_idx in range(self.model.n_layers):
            key = f"layer_{layer_idx}"
            if key not in cache.residual_stream:
                probs.append(0.0)
                continue
            hidden = cache.residual_stream[key][:, token_position, :].unsqueeze(0)
            with torch.no_grad():
                normed = self._ln_f(hidden)
                layer_logits = self._unembed(normed)[0, 0, :]
                prob = torch.softmax(layer_logits, dim=-1)[target_id].item()
            probs.append(prob)

        return np.array(probs)


class LogitLensResult:
    def __init__(self, prompt, per_layer, final_top_tokens, token_position):
        self.prompt = prompt
        self.per_layer = per_layer
        self.final_top_tokens = final_top_tokens
        self.token_position = token_position

    def __repr__(self):
        lines = [f"LogitLensResult for: '{self.prompt}'", ""]
        for entry in self.per_layer:
            top = entry["top_tokens"][0]
            lines.append(f"  Layer {entry['layer']:2d} → '{top[0].strip()}' ({top[1]:.3f})")
        lines.append(f"\n  Final   → '{self.final_top_tokens[0][0].strip()}' ({self.final_top_tokens[0][1]:.3f})")
        return "\n".join(lines)
