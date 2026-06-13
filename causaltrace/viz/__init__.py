"""
CausalTrace: Mechanistic Interpretability Toolkit for Transformer Models

Trace which attention heads and MLP layers are causally responsible
for factual outputs using activation patching, logit lens, and causal scrubbing.
"""

__version__ = "0.1.0"
__author__ = "CausalTrace Contributors"

from causaltrace.core.model_wrapper import ModelWrapper
from causaltrace.core.patching import ActivationPatcher
from causaltrace.core.logit_lens import LogitLens
from causaltrace.core.causal_scrubbing import CausalScrubber

__all__ = [
    "ModelWrapper",
    "ActivationPatcher",
    "LogitLens",
    "CausalScrubber",
]
