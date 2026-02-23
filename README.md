# RoPE vs PoPE

This repository compares **RoPE** (Rotary Position Embedding) and **PoPE** (Polar Coordinate Position Embedding) for position encoding in GPT-style transformers. It is built on top of [nanoGPT](https://github.com/karpathy/nanoGPT).

> **Using nanoGPT?** For the original nanoGPT setup (install, Docker, quick start, finetuning, etc.), see **[README_NANGPT.md](README_NANGPT.md)**.

---

## Prerequisites

- Python 3.10+
- pip

The Shakespeare config runs on CPU by default, so a GPU is optional for the quick start.

---

## Quick start

1. **Prepare data:**
   ```sh
   python data/shakespeare_char/prepare.py
   ```

2. **Train** (choose one of the commands in "How to use" below). You must train before sampling.

3. **Sample** (after training completes):
   ```sh
   python sample.py config/train_shakespeare_char.py --start="To be or not to be" --num_samples=3 --max_new_tokens=200
   ```

---

## How to use

**Installation**
```sh
python -m venv popenv
source popenv/bin/activate   # macOS/Linux
# popenv\Scripts\activate   # Windows
pip install -r requirements.txt
```

**Train with PoPE:**
```sh
python train.py config/train_shakespeare_char.py --use_pope=True
```

**Train with RoPE:**
```sh
python train.py config/train_shakespeare_char.py --use_rope=True
```

**Train with learned positional embeddings (default):**
```sh
python train.py config/train_shakespeare_char.py
```

The flags are mutually exclusive — setting both `use_rope=True` and `use_pope=True` will raise an assertion error.

---

## PoPE Implementation Summary

### What is PoPE?

**PoPE** (Polar Coordinate Position Embedding) from [arxiv.org/abs/2509.10534](https://arxiv.org/abs/2509.10534) decouples the "what" (content) from the "where" (position) in attention. RoPE entangles these via a phase interaction term φ_k - φ_q, while PoPE eliminates this confound.

### Key differences from RoPE

| Aspect | RoPE | PoPE |
|--------|------|------|
| Frequencies | d/2 (operates on pairs) | d (operates on individual elements) |
| Content encoding | Raw values influence both magnitude and phase | Softplus for non-negative magnitudes only |
| Position encoding | Rotation + content-dependent phase shift | Pure positional rotation, no content interaction |
| Learnable params | None | Phase bias δ_c per head per frequency |

### Core equation (Eq 6 from paper)

```
a_ts = Σ_c μ_q_c · μ_k_c · cos((s - t) · θ_c + δ_c)
```

where μ = softplus(x), θ_c = base^{-(c-1)/d}, and δ_c is learnable bias clamped to [-2π, 0].

### Efficient computation (Eq 10)

The attention score is computed as Re[q^H k] using Cartesian form:

```
att = (q_real @ k_real^T) + (q_imag @ k_imag^T)
```

This avoids materializing complex numbers while being faithful to the paper.

### Code difference (attention in `model.py`)

**RoPE:** Rotate q, k in place, then standard attention.
```python
cos, sin = self.rope(q, seq_len=T)
q, k = RoPE.apply_rotary_pos_emb(q, k, cos, sin)
att = (q @ k.transpose(-2, -1)) * scale
```

**PoPE:** Transform to polar form, then two matmuls for Re[q^H k].
```python
q_real, q_imag, k_real, k_imag = self.pope(q, k, T)
att = (q_real @ k_real.transpose(-2,-1) + q_imag @ k_imag.transpose(-2,-1)) * scale
```

---

## Note on Flash Attention

PoPE uses manual attention computation (two matmuls for Re[q^H k]) since the complex-valued dot product structure differs from standard attention. RoPE and learned embeddings continue to use Flash Attention when available. For small-GPU experiments this is perfectly fine — the paper's authors also wrote a custom Triton kernel for production use.

---

## nanoGPT setup

For full nanoGPT documentation (install, Docker, reproducing GPT-2, finetuning, sampling, troubleshooting), see **[README_NANGPT.md](README_NANGPT.md)**.
