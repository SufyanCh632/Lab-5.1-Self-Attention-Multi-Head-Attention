# Lab 5.1 — Self-Attention & Multi-Head Attention

## 📌 Project Overview

This project demonstrates the implementation of **Self-Attention, Scaled Dot-Product Attention, and Multi-Head Attention** using Python and PyTorch.

Attention mechanisms are fundamental components of Transformer architectures. They allow neural networks to identify relationships between tokens in a sequence and assign different levels of importance to each token.

This lab explores how attention scores are calculated, how attention weights are generated, and how multiple attention heads process information.

## 🎯 Objectives

- Understand the concept of Self-Attention.
- Implement Scaled Dot-Product Attention from scratch.
- Calculate attention scores using Query, Key, and Value matrices.
- Apply the Softmax function to generate attention weights.
- Implement Multi-Head Attention using PyTorch.
- Understand attention output shapes.

## 🛠️ Technologies Used

- **Python** — Programming language
- **PyTorch** — Deep learning framework
- **Matplotlib** — Data visualization
- **Jupyter Notebook** — Development environment

## 📂 Project Structure

```text
Lab-5.1-Self-Attention/
│
├── Lab_5_1_Self_Attention.ipynb
├── requirements.txt
└── README.md
```

## 📚 Implementation

### 1. Toy Sequence Input

Generate a random input tensor representing a sequence of five tokens, each containing four features.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)

seq_len, d_model = 5, 4

x = torch.randn(1, seq_len, d_model)

print("Input Shape:", x.shape)
print("Input Tensor:\n", x)
```

**Output:**

```text
Input Shape: torch.Size([1, 5, 4])
```

### 2. Scaled Dot-Product Attention

Scaled Dot-Product Attention calculates attention scores using Query (Q), Key (K), and Value (V) matrices.

**Mathematical Formula:**

\[
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

**Implementation:**

```python
def attention(Q, K, V):

    scores = (Q @ K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)

    weights = F.softmax(scores, dim=-1)

    return weights @ V, weights


Q, K, V = x, x, x

out, w = attention(Q, K, V)

print("Attention output shape:", out.shape)
print("Attention weights shape:", w.shape)
```

**Output:**

```text
Attention output shape: torch.Size([1, 5, 4])
Attention weights shape: torch.Size([1, 5, 5])
```

### 3. Visualizing Self-Attention Weights (Optional)

Attention weights can be visualized using a heatmap to understand how tokens interact with one another.

```python
plt.matshow(w[0].detach().cpu().numpy())

plt.title("Self-Attention Weights")
plt.colorbar()
plt.show()
```

The heatmap represents attention relationships between five tokens.

### 4. Multi-Head Attention

Multi-Head Attention allows the model to learn different relationships between tokens simultaneously.

PyTorch provides a built-in `MultiheadAttention` module.

```python
mha = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=2,
    batch_first=True
)

out, weights
