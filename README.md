# 🔬 CausalTrace

**Mechanistic Interpretability Toolkit for Transformer Language Models**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io)

---

CausalTrace exposes what happens *inside* a transformer when it retrieves a fact.
Rather than treating a language model as a black box, this toolkit lets you trace
exactly **which attention heads and MLP layers are causally responsible** for a
given output — down to a single token.

It implements three core mechanistic interpretability techniques:

| Technique | What it answers |
|-----------|----------------|
| **Activation Patching** | "Which layer stores this fact?" |
| **Logit Lens** | "How does the prediction evolve layer by layer?" |
| **Causal Scrubbing** | "Is this circuit hypothesis sufficient to explain the behaviour?" |

---

## Why this matters

Most work on LLMs stays at the API surface. Mechanistic interpretability goes
deeper — it tries to reverse-engineer the actual *algorithm* a model uses to
produce an output. This is central to model safety, debugging factual errors,
and understanding emergent capabilities.

CausalTrace is a practical toolkit for that work, inspired by:
- Meng et al., [*Locating and Editing Factual Associations in GPT*](https://arxiv.org/abs/2202.05262) (NeurIPS 2022)
- nostalgebraist, [*interpreting GPT: the logit lens*](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) (2020)
- Chan et al., [*Causal Scrubbing*](https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing) (Redwood Research, 2022)

---

## Demo

```python
from causaltrace import ModelWrapper, ActivationPatcher, LogitLens

# Load any HuggingFace causal LM
model = ModelWrapper.from_pretrained("gpt2")

# Track how P(" Paris") evolves across layers
lens = LogitLens(model)
probs = lens.token_trajectory(
    "The Eiffel Tower is located in the city of",
    target_token=" Paris",
)
# → peaks at layer 17 (0.82) — the model "commits" at mid-to-late layers

# Identify which layers are causally responsible via activation patching
patcher = ActivationPatcher(model)
result = patcher.patch_sweep(
    prompt_clean="The Eiffel Tower is located in the city of",
    prompt_corrupted="xjqz Eiffel Tower is located in the city of",
    target_token=" Paris",
)
# → MLP layers 17–19 show recovery > 0.7; attention at layer 10 shows > 0.5
```

---

## Quick start

```bash
git clone https://github.com/yourusername/CausalTrace.git
cd CausalTrace
pip install -r requirements.txt

# Run the example script
python notebooks/01_factual_recall_trace.py

# Launch the interactive web UI
streamlit run web/app.py
```

The web UI lets you run all three analyses on any prompt without writing code:

```
http://localhost:8501
```

---

## Project structure

```
CausalTrace/
│
├── causaltrace/
│   ├── core/
│   │   ├── model_wrapper.py      # Unified hook interface over HuggingFace models
│   │   ├── patching.py           # Activation patching (causal tracing)
│   │   ├── logit_lens.py         # Per-layer vocabulary projection
│   │   └── causal_scrubbing.py   # Circuit hypothesis testing
│   │
│   └── viz/
│       └── plots.py              # Matplotlib visualizations
│
├── web/
│   └── app.py                    # Streamlit interactive UI
│
├── notebooks/
│   └── 01_factual_recall_trace.py   # End-to-end example: Eiffel Tower → Paris
│
├── tests/
├── requirements.txt
└── README.md
```

---

## Supported models

CausalTrace auto-detects the architecture and registers hooks appropriately.

| Architecture | Example models |
|---|---|
| GPT-2 family | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl` |
| GPT-NeoX family | `EleutherAI/gpt-neox-20b`, `EleutherAI/pythia-*` |
| LLaMA family | `meta-llama/Llama-2-7b-hf`, `meta-llama/Meta-Llama-3-8B` |
| Mistral family | `mistralai/Mistral-7B-v0.1` |

---

## Core concepts

### Activation patching

The key idea: if patching a component's activation from a "clean" run
into a "corrupted" run restores the target token's probability, that component
is causally responsible.

```
Clean run:     "The Eiffel Tower is in ___"  → P(" Paris") = 0.82
Corrupted run: "xjqz Eiffel Tower is in ___" → P(" Paris") = 0.03

Patch MLP@layer17:  → P(" Paris") = 0.79  (recovery = 0.94 ✓ causally important)
Patch MLP@layer3:   → P(" Paris") = 0.05  (recovery = 0.02 ✗ not responsible)
```

### Logit lens

At each layer, the residual stream is projected through the final LayerNorm and
unembedding matrix. This converts intermediate hidden states into probability
distributions, revealing how "sure" the model is at each depth.

### Causal scrubbing

A rigorous test for circuit hypotheses: given a claim "heads L5H2, L8H7 and
MLP@L17 implement indirect object identification", causal scrubbing resamples
all other activations from unrelated inputs. If performance is preserved, the
circuit hypothesis is sufficient.

---

## Extending CausalTrace

Adding a new architecture requires one dict entry in `ModelWrapper.SUPPORTED_ARCH`:

```python
ModelWrapper.SUPPORTED_ARCH["falcon"] = (
    "transformer.h",   # dotted path to layer list
    "self_attention",  # attribute name for attention module
    "mlp",             # attribute name for MLP module
)
```

---

## Roadmap

- [ ] Attention head–level patching (per-head recovery scores)
- [ ] Automatic circuit discovery (top-k components by recovery)
- [ ] TransformerLens backend integration
- [ ] Export results as JSON / CSV for downstream analysis
- [ ] Colab notebook
- [ ] Support for encoder-decoder models (T5, BART)

---

## Citation

If you use CausalTrace in your research, please cite the foundational work:

```bibtex
@article{meng2022rome,
  title   = {Locating and Editing Factual Associations in {GPT}},
  author  = {Meng, Kevin and Bau, David and Andonian, Alex and Belinkov, Yonatan},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2022}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built to understand what language models actually know — and where they know it.*
[README.md](https://github.com/user-attachments/files/28915567/README.md)
