# Plan: Improve Semantic ID Quality and Global Uniqueness

## Objective
Increase global semantic-ID uniqueness without harming semantic coherence, while keeping scope limited to the semantic-ID generator (not a full recommender).

## Guiding principles
- Balance uniqueness with reconstruction fidelity and interpretability.
- Prefer stable training dynamics over aggressive regularization.
- Add diagnostics before adding complex new objectives.

## Phase 1: Measurement and baselines
**Goal:** Make uniqueness measurable and comparable across runs.

1. **Define primary uniqueness metrics**
   - Global uniqueness ratio: number of distinct semantic ID sequences divided by total items.
   - Per-layer usage: fraction of active codebook entries per layer.
   - Per-layer entropy: entropy of codebook usage per layer.

2. **Define secondary quality metrics**
   - Reconstruction error distribution (mean, median, tails).
   - Residual norm ratios across layers.
   - Stability across seeds (variance of uniqueness metrics).

3. **Establish baseline runs**
   - Fix dataset split, seed, and batch size.
   - Record all metrics above to serve as a reference point.

## Phase 2: Stabilize quantization dynamics
**Goal:** Reduce collapse so deeper layers contribute meaningfully to uniqueness.

1. **Residual normalization**
   - Normalize residuals per layer to prevent later layers from seeing vanishing signal.
   - Track changes in per-layer coverage and entropy.

2. **Layer-wise training schedule**
   - Train early layers first, then gradually unlock deeper layers.
   - Compare uniqueness and reconstruction across schedules.

3. **Loss scaling consistency**
   - Ensure reconstruction and quantization terms are on stable scales.
   - Validate that changes do not inflate uniqueness at the cost of severe reconstruction drift.

## Phase 3: Codebook health and diversity
**Goal:** Prevent dead codebooks and improve coverage.

1. **Dead-code detection**
   - Define a threshold for inactivity per codebook entry.
   - Log inactive entries by layer.

2. **Codebook reinitialization policy**
   - Reinitialize persistently inactive entries using recent residual samples.
   - Keep a cooldown window to avoid oscillation.

3. **Diversity regularization (lightweight)**
   - Add a mild penalty that discourages overly peaked code usage distributions.
   - Monitor uniqueness gain vs reconstruction impact.

## Phase 4: Semantic alignment without full recommender
**Goal:** Improve interpretability and uniqueness by weak semantic supervision.

1. **Attribute-aligned regularization**
   - Use available metadata (genres, year) to encourage semantic grouping without collapsing IDs.
   - Evaluate with mutual information or class-conditional entropy metrics.

2. **Contrastive structure (optional, small scale)**
   - Introduce a small contrastive loss between items sharing metadata labels.
   - Monitor whether global uniqueness remains stable or improves.

## Phase 5: Global uniqueness target tuning
**Goal:** Make uniqueness a controllable, not accidental, property.

1. **Uniqueness target scheduling**
   - Set a target range for global uniqueness (not just maximize).
   - Increase target gradually across epochs.

2. **Uniqueness-aware early stopping**
   - Stop only when both uniqueness and reconstruction meet thresholds.
   - Avoid stopping on uniqueness alone.

## Deliverables
- A metrics dashboard/log report summarizing uniqueness, coverage, entropy, and reconstruction.
- A comparison table of uniqueness vs reconstruction across experiments.
- A short decision note on which combination yields the best trade-off.

## Success criteria
- Global uniqueness ratio increases relative to baseline.
- Per-layer coverage improves with no layer collapse.
- Reconstruction error does not degrade beyond an acceptable margin.
- Results are stable across multiple seeds.
