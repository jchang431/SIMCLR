# CIFAR-10 Data Splits

All splits are derived from the **CIFAR-10 training set** (50 000 images, 10 classes).
The CIFAR-10 test set (10 000 images, balanced 1 000/class) is always used unchanged for final evaluation.

Every `.npz` file contains three index arrays that index into the full CIFAR-10 training set:

| Key | Meaning |
|-----|---------|
| `labeled_idx` | Indices used as the **training set** for linear eval |
| `val_idx` / `val_indices` | Indices held out for **validation** during training |
| `unlabeled_idx` | Remaining indices (not used in linear eval — backbone is frozen) |

Seed is fixed at **42** across all splits for reproducibility.

---

## data/balanced/

Class-balanced splits. Each class gets an equal share of the labeled budget.
The validation set is always a stratified 5 000-sample hold-out (500/class for imbalanced; ~500/class for balanced).

| File | Config key (`data.percent`) | Labeled | Per class (approx) | Val |
|------|-----------------------------|---------|-------------------|-----|
| `cifar10_split_0p01_seed42.npz` | `0.01` | 445 | ~44–45 | 5 000 |
| `cifar10_split_0p05_seed42.npz` | `0.05` | 2 244 | ~223–226 | 5 000 |
| `cifar10_split_0p1_seed42.npz`  | `0.1`  | 4 496 | ~447–452 | 5 000 |
| `cifar10_split_0p25_seed42.npz` | `0.25` | 11 247 | ~1 118–1 131 | 5 000 |
| `cifar10_split_0p5_seed42.npz`  | `0.5`  | 22 497 | ~2 236–2 262 | 5 000 |
| `cifar10_split_1p0_seed42.npz`  | `1.0`  | 45 000 | ~4 473–4 524 | 5 000 |

**Usage** — set `data.percent` in `config/simclr_config.yml` (leave `data.imbalanced_split: null`):

```yaml
data:
  percent: 0.1        # 10% labeled budget
  imbalanced_split: null
```

---

## data/imbalanced/

Imbalanced splits where each class receives a **different** number of labeled examples.
The validation set is always balanced at **500 per class (5 000 total)**.

### Difficulty-weighted splits

Labels are allocated so that classes the model finds hardest get the **most** examples.
Hard classes: bird, cat, dog (700 labels each at full scale).
Easy classes: automobile, ship, truck (~266–267 labels each at full scale).

| File | Config key (`data.imbalanced_split`) | Labeled | Val |
|------|--------------------------------------|---------|-----|
| `cifar10_imb_difficulty_seed42.npz` | `difficulty` | 4 500 | 5 000 |
| `cifar10_imb_difficulty_1pct_seed42.npz` | `difficulty_1pct` | 450 | 5 000 |

Per-class breakdown (full scale):

| Class | difficulty | difficulty_1pct |
|-------|-----------|-----------------|
| airplane | 400 | 40 |
| automobile | 266 | 26 |
| bird | 700 | 70 |
| cat | 700 | 70 |
| deer | 400 | 40 |
| dog | 700 | 70 |
| frog | 400 | 40 |
| horse | 400 | 40 |
| ship | 267 | 27 |
| truck | 267 | 27 |

---

### Long-tail splits — Imbalance Factor 10 (IF=10)

Labels follow an **exponential decay** from the most-common class (airplane) to the rarest (truck).
The most-common class has **10×** more labels than the rarest class.

| File | Config key | Labeled | Val |
|------|------------|---------|-----|
| `cifar10_imb_lt_if10_seed42.npz` | `lt_if10` | 4 492 | 5 000 |
| `cifar10_imb_lt_if10_1pct_seed42.npz` | `lt_if10_1pct` | 446 | 5 000 |
| `cifar10_imb_lt_if10_25pct_seed42.npz` | `lt_if10_25pct` | 11 236 | 5 000 |

Per-class breakdown:

| Class | lt_if10 | lt_if10_1pct | lt_if10_25pct |
|-------|---------|--------------|---------------|
| airplane | 1 100 | 110 | 2 750 |
| automobile | 851 | 85 | 2 129 |
| bird | 659 | 65 | 1 648 |
| cat | 510 | 51 | 1 276 |
| deer | 395 | 39 | 988 |
| dog | 306 | 30 | 765 |
| frog | 236 | 23 | 592 |
| horse | 183 | 18 | 458 |
| ship | 142 | 14 | 355 |
| truck | 110 | 11 | 275 |

---

### Long-tail splits — Imbalance Factor 50 (IF=50)

Same exponential structure as IF=10 but with a much steeper tail.
The most-common class (airplane) has **50×** more labels than the rarest (truck).

| File | Config key | Labeled | Val |
|------|------------|---------|-----|
| `cifar10_imb_lt_if50_seed42.npz` | `lt_if50` | 4 476 | 5 000 |

Per-class breakdown:

| Class | lt_if50 |
|-------|---------|
| airplane | 1 600 |
| automobile | 1 035 |
| bird | 670 |
| cat | 434 |
| deer | 281 |
| dog | 182 |
| frog | 117 |
| horse | 76 |
| ship | 49 |
| truck | 32 |

---

## Quick comparison: balanced vs imbalanced (same labeled budget ~4 500)

| Split | Type | airplane | truck | Ratio |
|-------|------|----------|-------|-------|
| `0p1` (balanced) | Balanced | 452 | 450 | 1.0× |
| `difficulty` | Difficulty-weighted | 400 | 267 | 1.5× |
| `lt_if10` | Long-tail IF=10 | 1 100 | 110 | 10× |
| `lt_if50` | Long-tail IF=50 | 1 600 | 32 | 50× |

---

## Usage in code

**Via config file** (`config/simclr_config.yml`):

```yaml
data:
  dataset: CIFAR
  percent: 1               # used only when imbalanced_split is null
  imbalanced_split: lt_if50   # or null for balanced splits
```

**Via CLI** (overrides the config value):

```bash
python3 main.py --mode linear --config config/simclr_config.yml --imb lt_if50 -t --viz
```

Valid `--imb` / `imbalanced_split` values:
`difficulty` · `difficulty_1pct` · `lt_if10` · `lt_if10_1pct` · `lt_if10_25pct` · `lt_if50`
