# Phase 4 Plan: Semantic Alignment Without a Full Recommender

## Purpose

Increase interpretability and stabilize uniqueness by lightly aligning semantic IDs with item metadata (genres, year) without adding a generative recommender.

## What to change in this codebase

- Add an optional weak-supervision loss using metadata available from MovieLens item features.
- Add metadata-conditioned diagnostics that measure how semantic IDs distribute across genres and years.

## Where it fits

- Data loading and metadata extraction in [data/loader.py](data/loader.py).
- Training loop augmentation in [train_rq_vae.py](train_rq_vae.py).
- Optional analysis scripts under [scripts/visualizations](scripts/visualizations).

## Implementation details

### 1) Return metadata labels from loader

In [data/loader.py](data/loader.py), add an optional flag to return labels:

```python
def load_movie_lens(..., return_labels=False):
	...
	if return_labels:
		labels = data["genre:token_seq"].astype("category").cat.codes.values
		return embeddings, torch.tensor(labels)
	return embeddings
```

### 2) Pass labels into training

In [train_rq_vae.py](train_rq_vae.py):

```python
data, labels = load_movie_lens(..., return_labels=True)
train_loader = DataLoader(list(zip(data, labels)), batch_size=...)
```

Update the training loop to unpack (batch, label):

```python
for batch, label in train_loader:
	result = model(batch.to(device))
```

### 3) Auxiliary alignment loss

Add a lightweight cohesion loss using cosine similarity between items with the same label:

```python
z = model.encode(batch.to(device))
same = label[:, None] == label[None, :]
sim = torch.nn.functional.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)
align_loss = (1 - sim[same]).mean()
loss = loss + align_weight * align_loss
```

### 4) Alignment diagnostics

Compute conditional entropy per label after each validation step:

```python
for c in labels.unique():
	ids_c = sem_ids[labels == c]
	counts = torch.bincount(ids_c[:, layer], minlength=codebook_size).float()
	probs = counts / counts.sum().clamp_min(1.0)
	entropy_c = -(probs * (probs + 1e-9).log()).sum()
```

## Required adaptations

- Pass metadata labels alongside embeddings in the training loop.
- Compute an auxiliary loss that encourages within-genre cohesion and cross-genre separation.
- Ensure the auxiliary loss is weighted lightly to preserve reconstruction quality.

## What to append

- A metadata alignment report per layer (conditional entropy or mutual information proxy).
- Small qualitative plots to confirm that semantic IDs align with genres or release years.

## Why this helps

- Weak supervision anchors semantic IDs to human-interpretable attributes.
- Alignment improves explainability while still allowing uniqueness to increase.
- It provides a non-recommender way to validate semantic IDs beyond reconstruction loss.

## Deliverables

- Optional metadata-aligned loss in training.
- Standardized alignment metrics and plots.
- A short report that compares uniqueness and alignment before and after changes.
