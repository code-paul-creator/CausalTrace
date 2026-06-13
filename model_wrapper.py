"""
ModelWrapper: Unified interface over HuggingFace transformer models
with hooks for extracting residual stream, attention patterns, and MLP outputs.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ActivationCache:
    """Stores activations captured during a forward pass."""
    residual_stream: Dict[str, torch.Tensor] = field(default_factory=dict)
    attention_patterns: Dict[str, torch.Tensor] = field(default_factory=dict)
    mlp_outputs: Dict[str, torch.Tensor] = field(default_factory=dict)
    attn_outputs: Dict[str, torch.Tensor] = field(default_factory=dict)

    def clear(self):
        self.residual_stream.clear()
        self.attention_patterns.clear()
        self.mlp_outputs.clear()
        self.attn_outputs.clear()


class ModelWrapper:
    """
    Wraps a HuggingFace causal LM with hooks to capture internal activations.

    Supports GPT-2, GPT-J, LLaMA, Mistral, and compatible architectures.

    Example
    -------
    >>> model = ModelWrapper.from_pretrained("gpt2")
    >>> with model.trace("The Eiffel Tower is located in"):
    ...     logits, cache = model.run()
    >>> cache.residual_stream["layer_6"].shape
    torch.Size([1, 8, 768])
    """

    SUPPORTED_ARCH = {
        "gpt2": ("transformer.h", "attn", "mlp"),
        "gpt_neox": ("gpt_neox.layers", "attention", "mlp"),
        "llama": ("model.layers", "self_attn", "mlp"),
        "mistral": ("model.layers", "self_attn", "mlp"),
    }

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        arch_key: str = "gpt2",
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.arch_key = arch_key
        self.cache = ActivationCache()
        self._hooks: List = []
        self._prompt: Optional[str] = None

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cpu",
        **kwargs,
    ) -> "ModelWrapper":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        arch_key = cls._detect_arch(model)
        return cls(model, tokenizer, arch_key=arch_key, device=device)

    @staticmethod
    def _detect_arch(model: nn.Module) -> str:
        name = type(model).__name__.lower()
        if "gpt2" in name:
            return "gpt2"
        if "neox" in name:
            return "gpt_neox"
        if "llama" in name:
            return "llama"
        if "mistral" in name:
            return "mistral"
        return "gpt2"  # fallback

    def trace(self, prompt: str) -> "ModelWrapper":
        """Context manager: set the prompt and register activation hooks."""
        self._prompt = prompt
        return self

    def __enter__(self):
        self._register_hooks()
        return self

    def __exit__(self, *_):
        self._remove_hooks()

    def _register_hooks(self):
        """Register forward hooks on every transformer layer."""
        layers_path, attn_name, mlp_name = self.SUPPORTED_ARCH[self.arch_key]

        # Resolve layer list via dotted path
        layers = self.model
        for part in layers_path.split("."):
            layers = getattr(layers, part)

        for layer_idx, layer in enumerate(layers):
            key = f"layer_{layer_idx}"

            # Residual stream (input to each layer)
            def make_residual_hook(k):
                def hook(module, inp, out):
                    self.cache.residual_stream[k] = inp[0].detach()
                return hook

            h = layer.register_forward_hook(make_residual_hook(key))
            self._hooks.append(h)

            # Attention pattern
            attn_module = getattr(layer, attn_name, None)
            if attn_module is not None:
                def make_attn_hook(k):
                    def hook(module, inp, out):
                        # out is typically (attn_output, attn_weights, ...)
                        if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                            self.cache.attention_patterns[k] = out[1].detach()
                        self.cache.attn_outputs[k] = out[0].detach()
                    return hook
                h = attn_module.register_forward_hook(make_attn_hook(key))
                self._hooks.append(h)

            # MLP output
            mlp_module = getattr(layer, mlp_name, None)
            if mlp_module is not None:
                def make_mlp_hook(k):
                    def hook(module, inp, out):
                        self.cache.mlp_outputs[k] = out.detach()
                    return hook
                h = mlp_module.register_forward_hook(make_mlp_hook(key))
                self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def run(self, prompt: Optional[str] = None) -> Tuple[torch.Tensor, ActivationCache]:
        """
        Run a forward pass and return (logits, cache).

        Parameters
        ----------
        prompt : str, optional
            If provided, overrides the prompt set in `trace()`.
        """
        self.cache.clear()
        text = prompt or self._prompt
        if text is None:
            raise ValueError("Provide a prompt via trace() or run(prompt=...)")

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        return outputs.logits, self.cache

    def top_tokens(self, logits: torch.Tensor, k: int = 10) -> List[Tuple[str, float]]:
        """Return top-k predicted tokens with probabilities for the last position."""
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top = torch.topk(probs, k)
        return [
            (self.tokenizer.decode([idx.item()]), prob.item())
            for idx, prob in zip(top.indices, top.values)
        ]

    @property
    def n_layers(self) -> int:
        layers_path = self.SUPPORTED_ARCH[self.arch_key][0]
        layers = self.model
        for part in layers_path.split("."):
            layers = getattr(layers, part)
        return len(layers)

    @property
    def d_model(self) -> int:
        return self.model.config.hidden_size
