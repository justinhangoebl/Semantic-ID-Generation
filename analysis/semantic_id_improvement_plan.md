# Plan: Improve Semantic ID Quality and Global Uniqueness

## Objective

Increase global semantic-ID uniqueness without harming semantic coherence, while keeping scope limited to the semantic-ID generator (not a full recommender).

## Guiding principles

- Balance uniqueness with reconstruction fidelity and interpretability.
- Prefer stable training dynamics over aggressive regularization.
- Add diagnostics before adding complex new objectives.

## Phase 1: Measurement and baselines

**Goal:** Make uniqueness measurable and comparable across runs, with minimal code to compute core metrics.

1. **Define primary uniqueness metrics**
   - Global uniqueness ratio: number of distinct semantic ID sequences divided by total items.
   - Per-layer usage: fraction of active codebook entries per layer.
   - Per-layer entropy: entropy of codebook usage per layer.

   Example metric computation (PyTorch):

   ```python
   # sem_ids: Tensor [num_items, num_layers]
   unique_ratio = torch.unique(sem_ids, dim=0).shape[0] / sem_ids.shape[0]

   # per-layer usage
   per_layer_usage = []
   per_layer_entropy = []
   for layer in range(sem_ids.shape[1]):
       ids = sem_ids[:, layer]
       counts = torch.bincount(ids, minlength=codebook_size).float()
       probs = counts / counts.sum().clamp_min(1.0)
       usage = (counts > 0).float().mean()
       entropy = -(probs * (probs + 1e-9).log()).sum()
       per_layer_usage.append(usage)
       per_layer_entropy.append(entropy)
   ```

2. **Define secondary quality metrics**
   - Reconstruction error distribution (mean, median, tails).
   - Residual norm ratios across layers.
   - Stability across seeds (variance of uniqueness metrics).

3. **Establish baseline runs**
   - Fix dataset split, seed, and batch size.
   - Record all metrics above to serve as a reference point.

## Phase 2: Stabilize quantization dynamics

**Goal:** Reduce collapse so deeper layers contribute meaningfully to uniqueness. Keep changes small and test one at a time.

1. **Residual normalization**
   - Normalize residuals per layer to prevent later layers from seeing vanishing signal.
   - Track changes in per-layer coverage and entropy.

   Example (inside the residual loop):

   ```python
   # current_residuals: [batch, latent_dim]
   current_residuals = torch.nn.functional.normalize(current_residuals, dim=-1)
   ```

2. **Layer-wise training schedule**
   - Train early layers first, then gradually unlock deeper layers.
   - Compare uniqueness and reconstruction across schedules.

   Example schedule (conceptual):

   ```python
   # epoch-based unlock
   unlocked_layers = 1 + epoch // steps_per_layer
   unlocked_layers = min(unlocked_layers, num_layers)
   ```

3. **Loss scaling consistency**
   - Ensure reconstruction and quantization terms are on stable scales.
   - Validate that changes do not inflate uniqueness at the cost of severe reconstruction drift.

   Example (consistent reductions):

   ```python
   recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")
   commit_loss = quant_loss.mean()
   total_loss = recon_weight * recon_loss + commit_weight * commit_loss
   ```

## Phase 3: Codebook health and diversity

**Goal:** Prevent dead codebooks and improve coverage with lightweight interventions.

1. **Dead-code detection**
   - Define a threshold for inactivity per codebook entry.
   - Log inactive entries by layer.

   Example (count last N steps):

   ```python
   # usage_counts: [num_layers, codebook_size]
   inactive_mask = usage_counts < dead_threshold
   ```

2. **Codebook reinitialization policy**
   - Reinitialize persistently inactive entries using recent residual samples.
   - Keep a cooldown window to avoid oscillation.

   Example (reseed from residuals):

   ```python
   with torch.no_grad():
       dead_ids = inactive_mask[layer].nonzero(as_tuple=True)[0]
       if dead_ids.numel() > 0:
           sampled = residuals[torch.randint(0, residuals.size(0), (dead_ids.numel(),))]
           codebook.weight[dead_ids] = sampled
   ```

3. **Diversity regularization (lightweight)**
   - Add a mild penalty that discourages overly peaked code usage distributions.
   - Monitor uniqueness gain vs reconstruction impact.

   Example (entropy bonus):

   ```python
   entropy_bonus = per_layer_entropy[layer]
   total_loss = total_loss - entropy_weight * entropy_bonus
   ```

## Phase 4: Semantic alignment without full recommender

**Goal:** Improve interpretability and uniqueness by weak semantic supervision and small auxiliary losses.

1. **Attribute-aligned regularization**
   - Use available metadata (genres, year) to encourage semantic grouping without collapsing IDs.
   - Evaluate with mutual information or class-conditional entropy metrics.

   Example (class-conditional entropy proxy):

   ```python
   # ids: [num_items], labels: [num_items]
   for c in labels.unique():
       ids_c = ids[labels == c]
       counts = torch.bincount(ids_c, minlength=codebook_size).float()
       probs = counts / counts.sum().clamp_min(1.0)
       entropy_c = -(probs * (probs + 1e-9).log()).sum()
   ```

2. **Contrastive structure (optional, small scale)**
   - Introduce a small contrastive loss between items sharing metadata labels.
   - Monitor whether global uniqueness remains stable or improves.

   Example (pairwise cosine):

   ```python
   # z: [batch, latent_dim], same_label_mask: [batch, batch]
   sim = torch.nn.functional.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)
   pos_loss = (1 - sim[same_label_mask]).mean()
   total_loss = total_loss + contrastive_weight * pos_loss
   ```

## Phase 5: Global uniqueness target tuning

**Goal:** Make uniqueness a controllable, not accidental, property by shaping training signals.

1. **Uniqueness target scheduling**
   - Set a target range for global uniqueness (not just maximize).
   - Increase target gradually across epochs.

   Example (target schedule):

   ```python
   target = min_target + (max_target - min_target) * epoch / max_epochs
   uniqueness_penalty = torch.relu(target - unique_ratio)
   total_loss = total_loss + uniq_weight * uniqueness_penalty
   ```

2. **Uniqueness-aware early stopping**
   - Stop only when both uniqueness and reconstruction meet thresholds.
   - Avoid stopping on uniqueness alone.

   Example (dual-threshold stop):

   ```python
   if unique_ratio >= uniq_threshold and recon_loss <= recon_threshold:
       should_stop = True
   ```

## Deliverables

- A metrics dashboard/log report summarizing uniqueness, coverage, entropy, and reconstruction.
- A comparison table of uniqueness vs reconstruction across experiments.
- A short decision note on which combination yields the best trade-off.

## Success criteria

- Global uniqueness ratio increases relative to baseline.
- Per-layer coverage improves with no layer collapse.
- Reconstruction error does not degrade beyond an acceptable margin.
- Results are stable across multiple seeds.
