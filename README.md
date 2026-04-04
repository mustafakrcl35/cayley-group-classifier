# Cayley Table Group Classifier

Determines whether a given Cayley table represents a **cyclic group** using both algebraic methods and **machine learning (binary classification)**.

## Project Structure

```
cayley_group_classifier/
├── group_classifier.py          # Algebraic analysis (group axioms, cyclic test)
├── dataset_generator.py         # Cayley table dataset generator
├── feature_extraction.py        # Feature extraction pipeline
├── experiment_random_forest.py  # Random Forest experiment
├── test_classifier.py           # Unit tests
├── results/                     # Experiment results
│   ├── experiment_results.json  # All metrics (JSON)
│   ├── dataset.csv              # Generated dataset
│   ├── confusion_matrix.png     # Confusion matrix visualization
│   ├── roc_curve.png            # ROC curve
│   ├── feature_importance.png   # Feature importance ranking
│   └── distributions.png        # Class and order distributions
└── README.md
```

---

## Experiment: Binary Classification of Cyclic Groups

### Objective

Classify whether the group represented by a Cayley table is **cyclic or not** using machine learning.

### Dataset

The dataset is synthetically generated using `dataset_generator.py`:

| Category | Group Types | Sample Count |
|----------|-------------|:------------:|
| **Cyclic** | Z_2, Z_3, ..., Z_20 | 950 |
| **Non-Cyclic** | V_4, S_3, D_3-D_10, Q_8, Z_2xZ_2, Z_2xZ_4, ... | 1100 |
| **Total** | 41 unique group types | **2050** |

For each group structure, 50 different **random relabelings** are applied to produce isomorphic but visually distinct tables.

### Features

20 structural features are extracted from each Cayley table:

| Category | Features |
|----------|----------|
| **Structural** | order, is_symmetric, symmetry_ratio |
| **Element Orders** | max/min/mean/std_element_order, max_order_ratio, num_generators, generator_ratio, num_distinct_orders, order_entropy |
| **Statistical** | diagonal_identity_count, trace_value, latin_square, row_uniformity |
| **Topological** | num_involutions, self_inverse_ratio, num_subgroup_orders, euler_phi_ratio |

### Model: Random Forest

Hyperparameter optimization was performed via GridSearchCV (5-fold Stratified CV):

| Parameter | Value |
|-----------|:-----:|
| n_estimators | 50 |
| max_depth | 5 |
| min_samples_split | 2 |
| min_samples_leaf | 1 |

### Results

| Metric | Value |
|--------|:-----:|
| **Accuracy** | 1.0000 |
| **Precision** | 1.0000 |
| **Recall** | 1.0000 |
| **F1 Score** | 1.0000 |
| **AUC-ROC** | 1.0000 |

#### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

#### ROC Curve

![ROC Curve](results/roc_curve.png)

#### Feature Importance

![Feature Importance](results/feature_importance.png)

#### Class Distribution

![Distributions](results/distributions.png)

### Discussion

The model achieved 100% accuracy. This result is due to the extracted features (especially `max_order_ratio` and `euler_phi_ratio`) directly encoding the cyclic group property. By definition, a cyclic group contains at least one element of order *n*, which guarantees `max_order_ratio = 1.0`.

This demonstrates the power of well-designed hand-crafted features. Future experiments may include:
- Classification using only raw table data (flattened table) without structural features
- Repeating experiments with more challenging feature subsets
- Testing deep learning models (CNN, GNN) for structural learning

---

## Algebraic Analysis Tool

`group_classifier.py` can also be used as a standalone algebraic analysis tool:

```python
from group_classifier import CayleyTableAnalyzer

table = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0],
]

analyzer = CayleyTableAnalyzer(table, elements=["e", "a", "b", "c"])
analyzer.full_report()
```

Checks performed: closure, identity element, inverses, associativity, cyclic test, abelian check, subgroups, group identification.

## Installation and Usage

```bash
# Requirements
pip install numpy pandas scikit-learn matplotlib seaborn

# Run algebraic analysis examples
python group_classifier.py

# Run ML experiment
python experiment_random_forest.py

# Run tests
python test_classifier.py
```

## Requirements

- Python 3.10+
- numpy, pandas, scikit-learn, matplotlib, seaborn

## License

MIT
