# varsplit

**Protein variant-aware train/test splitting for machine learning methods.**

---
*Supports method validation that is approriate for real-world scenarios. You can split by position or mutation, as well as by mutational load. For example, if you want to test how well your model translates single-site variant data to multi-site variant predictions, you can split by mutational load. If you would like to know how well a model performs in generalizing data to new positions, you would want to split by position so there is no leakage between testing and training.* 

*This package aims to be an easy way to implement smart splitting strategies for protein engineering models.*

---

Current Major Limitations
  - Only supports sets built around one WT
  - Extensive testing not performed for various input files. May not be fully robust
    
Planned Additions
  - Splitting based upon structural features
  - clustering support for dealing with sets of variable sequences (ie not all variants of one protein)

---

## Installation

**Into an existing environment:**
```bash
pip install git+https://github.com/TCoulth/varsplit.git
```

Dependencies (numpy, pandas, matplotlib) are installed automatically.

**From source (for development):**
```bash
git clone https://github.com/TCoulth/varsplit.git
cd varsplit
pip install -e .
```

---

## Examples

The `examples/` folder contains worked notebooks using real DMS datasets.

| Notebook | Dataset | Variants | Demonstrates |
|----------|---------|----------|--------------|
| [`CBPA2_HUMAN_Tsuboyama_2023_1O6X_example.ipynb`](examples/CBPA2_HUMAN_Tsuboyama_2023_1O6X_example.ipynb) | CUE domain of ubiquitin-binding protein A ([Tsuboyama et al. 2023](https://www.nature.com/articles/s41586-023-06328-6)) | 1,357 singles + 711 doubles | All five strategies with visualizations |

Each notebook includes the dataset and runs end-to-end with no additional downloads required.

---

## Quick start

```python
from varsplit import train_test_split, KFold, VarSplit

# Functional API -- mirrors sklearn
train_idx, test_idx = train_test_split(df, strategy="positional", random_state=42)
train_df = df.iloc[train_idx]
test_df  = df.iloc[test_idx]

# Cross-validation
for train_idx, test_idx in KFold(n_splits=5, strategy="order").split(df):
    ...

# Reusable configured instance
vs = VarSplit(mutation_col="mutant", fitness_col="score")
train_idx, test_idx = vs.train_test_split(df, strategy="fitness")
```

---

## Input format

A pandas DataFrame with a mutation string column and optionally a fitness column:

```python
df = pd.DataFrame({
    "mutations": ["A23V", "G105S", "A23V:G105S", "A23V:G105S:L200F"],
    "score":     [0.42,   0.81,    0.63,          0.95],
})
```

**Supported mutation string formats:**

| Format | Example | Notes |
|--------|---------|-------|
| `standard` | `A23V`, `A23V:G105S` | Default. Colon-separated multi-mutants. |
| `hgvs` | `p.Ala23Val` | HGVS protein notation. |
| `mavedb` | `p.[Ala23Val;Gly105Ser]` | MaveDB bracket notation. |
| `infer` | — | Auto-detects from data. |

Multi-mutant delimiters: `:` `,` `/` `+` `;` or space. Wildtype (`""`, `None`, `"WT"`) → empty set → always assigned to train.

---

## Choosing a strategy

| Strategy | Train | Test | Question being asked |
|----------|-------|------|----------------------|
| `random` | Random sample | Random sample | Baseline |
| `positional` | Variants at training positions | Variants touching held-out positions | Can the model generalize to new positions? |
| `mutational` | Variants with seen substitutions | Variants with unseen (pos, AA) pairs | Can the model generalize to new amino acids at known positions? |
| `order` | Singles (or up to order N) | Higher-order mutants | Can the model predict higher-order mutants from lower-order data? |
| `fitness` | Low/mid fitness variants | High (or low) fitness variants | Can the model extrapolate toward high-fitness variants? |

The gap between `random` and any other strategy reflects how well the model actually generalizes in that dimension.

---

## API reference

### `train_test_split`

```python
train_idx, test_idx = train_test_split(
    df,                        # pandas DataFrame
    test_size=0.2,             # fraction for test set
    strategy="random",         # split strategy
    random_state=None,         # random seed
    mutation_col="mutations",  # mutation string column
    fmt="standard",            # mutation format
    fitness_col=None,          # fitness column (required for strategy="fitness")
    verbose=True,              # print split summary
    **kwargs,                  # strategy-specific options (see below)
)
```

Returns `(train_indices, test_indices)` as numpy arrays.

---

### `KFold`

```python
kf = KFold(
    n_splits=5,
    strategy="random",
    random_state=None,
    mutation_col="mutations",
    fmt="standard",
    fitness_col=None,
)

for train_idx, test_idx in kf.split(df):
    ...
```

For `strategy="order"`, folds equal the number of unique mutation orders, not `n_splits`.

---

### `VarSplit`

```python
vs = VarSplit(
    mutation_col="mutations",
    fmt="standard",
    fitness_col=None,
    reference=None,       # wildtype sequence string, or "consensus"
)

train_idx, test_idx = vs.train_test_split(df, strategy="positional")
for train_idx, test_idx in vs.kfold(df, n_splits=5, strategy="order"):
    ...
```

---

## Strategy options

### `positional`

```python
# Random position holdout
train_test_split(df, strategy="positional", test_size=0.2)

# Explicit positions
train_test_split(df, strategy="positional", held_out_positions={23, 105})
```

`test_size` controls fraction of *positions* held out, not variants. Multi-site variants go to test if they touch *any* held-out position.

---

### `mutational`

```python
# Random substitution holdout
train_test_split(df, strategy="mutational", test_size=0.2)

# Explicit substitutions
train_test_split(df, strategy="mutational", held_out_substitutions={(23, "V"), (105, "S")})
```

Holds out `(position, mutant_aa)` pairs. Positions may appear in both train and test. Warns if the dataset contains only single-site variants.

---

### `order`

```python
# Default: singles -> train, multimutants -> test
train_test_split(df, strategy="order")

# Custom threshold
train_test_split(df, strategy="order", train_max_order=2)
```

`test_size` is not used — split is determined by mutation order. Actual percentages are always reported. Wildtype always goes to train.

---

### `fitness`

```python
# Top 20% by fitness -> test
train_test_split(df, strategy="fitness", fitness_col="score", test_size=0.2)

# Absolute threshold
train_test_split(df, strategy="fitness", fitness_col="score", threshold=0.8)

# Bottom tail
train_test_split(df, strategy="fitness", fitness_col="score", test_size=0.1, upper_tail=False)
```

---

## Verbose output

All strategies print a summary by default. Protein-aware splits rarely produce exact `test_size` fractions, so knowing the actual split is important.

```
Positional split (method=random, 2/10 positions held out): train=312 (74.1%), test=109 (25.9%)
Order split (train_max_order=1): train=190 (45.2%), test=230 (54.8%)
  Order distribution: {1: 190, 2: 148, 3: 82}
Fitness split (quantile, top tail, cutoff=0.823): train=336 (79.8%), test=85 (20.2%)
  Train score range: [0.012, 0.821]
  Test score range:  [0.823, 0.998]
```

Suppress with `verbose=False`.

---

## Adding a custom strategy

1. Subclass `BaseSplitStrategy` in `varsplit/strategies/`:

```python
from varsplit.strategies.base import BaseSplitStrategy
import numpy as np

class MyStrategy(BaseSplitStrategy):

    def split(self, variants, test_size=0.2, random_state=None, **kwargs):
        # variants: list[frozenset[tuple[int, str, str]]]
        # each element: set of (position, wt_aa, mut_aa) tuples
        train_idx = ...
        test_idx  = ...
        self._validate_split(variants, train_idx, test_idx)
        return train_idx, test_idx
```

2. Register it:

```python
# varsplit/strategies/__init__.py
REGISTRY["my_strategy"] = MyStrategy
```

**New strategy vs. new option:** add a new strategy when the *unit being held out* changes. Add an option/method when only the *selection logic* changes (e.g. different ways to choose which positions to hold out).

---



## License

MIT
