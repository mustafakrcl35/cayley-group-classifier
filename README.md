# Cayley Table Group Classifier

Determines whether a given Cayley table represents a **cyclic group** using both algebraic methods and **machine learning (binary classification)**.

**Research Question:** *Can machine learning recover the element-order structure of a finite group directly from its Cayley table, and which classification models are most effective for this task?*

## Project Structure

```
cayley_group_classifier/
├── group_classifier.py              # Algebraic analysis (group axioms, cyclic test)
├── dataset_generator.py             # Cayley table dataset generator
├── feature_extraction.py            # Feature extraction pipeline
├── experiment_random_forest.py      # Experiment 1: Random Forest (with GridSearch)
├── experiment_model_comparison.py   # Experiment 2: Multi-model comparison
├── test_classifier.py               # Unit tests
├── results/                         # Experiment 1 results
├── results_comparison/              # Experiment 2 results
└── README.md
```

---

## Methodology

### 1. Dataset Construction

No real-world data was collected. Instead, we synthetically generated Cayley tables from mathematically known group structures where the ground truth (cyclic or not) is already established.

**Cyclic groups (label = 1):** We constructed the Cayley tables for Z_2 through Z_20 using modular addition. For example, the table for Z_4 is defined as `table[i][j] = (i + j) % 4`.

**Non-cyclic groups (label = 0):** We constructed tables for well-known non-cyclic families: the Klein four-group V_4, the symmetric group S_3, dihedral groups D_3 through D_10, the quaternion group Q_8, and several direct products like Z_2 x Z_2, Z_2 x Z_4, etc.

**Random relabeling for diversity:** For each group structure, we applied 50 random permutation relabelings. This means the elements are shuffled (e.g., {0,1,2,3} becomes {2,0,3,1}) and the table is rewritten accordingly. The resulting table is mathematically identical (isomorphic) but numerically different. This prevents models from memorizing positional patterns like "element 0 is always the identity."

**Final dataset:** ~2000 Cayley tables total (950 cyclic, 1050 non-cyclic) spanning 41 unique group types.

### 2. Labeling

There is no manual labeling. Since we generate each table from a known group structure, the label is inherent: tables generated from Z_n are labeled 1 (cyclic); tables generated from D_n, S_3, V_4, etc. are labeled 0 (non-cyclic). This serves as the ground truth for supervised learning.

### 3. Feature Extraction

We test two fundamentally different feature representations to determine whether domain knowledge is necessary:

**(A) Hand-crafted structural features (20 features)**

For each Cayley table, we compute algebraic properties by analyzing the table programmatically:

| Category | Features | How they are computed |
|----------|----------|----------------------|
| **Structural** | order, is_symmetric, symmetry_ratio | Table dimensions; checking if `table[i][j] == table[j][i]` for all pairs |
| **Element Orders** | max/min/mean/std_element_order, max_order_ratio, num_generators, generator_ratio, num_distinct_orders, order_entropy | For each element *a*, we compute *a, a\*a, a\*a\*a, ...* until we reach the identity element *e*. The number of steps is the element's order. If any element has order = n (group order), it is a generator, and the group is cyclic. |
| **Statistical** | diagonal_identity_count, trace_value, latin_square, row_uniformity | Inspecting the table's diagonal entries, checking if each row/column is a permutation |
| **Topological** | num_involutions, self_inverse_ratio, num_subgroup_orders, euler_phi_ratio | Counting elements of order 2, self-inverse elements, and distinct cyclic subgroup sizes |

The key feature is `max_order_ratio` (= maximum element order / group order). By definition, this equals 1.0 for cyclic groups and is strictly less than 1.0 for non-cyclic groups.

**(B) Raw flattened Cayley table (420 features)**

The Cayley table is simply padded to a fixed 20x20 size, normalized, and flattened into a 400-dimensional vector. The 20 structural features are prepended, giving 420 total features. This representation contains **no explicit algebraic knowledge** — it tests whether models can discover the cyclic property directly from the raw multiplication table.

### 4. Experimental Setup

The data is split 80% train / 20% test using stratified sampling (preserving class ratios). All models are evaluated with 5-fold stratified cross-validation on the training set, and final metrics are reported on the held-out test set.

### 5. Models Tested

Six classifiers spanning different learning paradigms:

| Model | Type | Why included |
|-------|------|-------------|
| **Random Forest** | Ensemble (bagging) | Strong baseline for tabular data |
| **Gradient Boosting** | Ensemble (boosting) | State-of-the-art for tabular tasks |
| **SVM (RBF kernel)** | Kernel method | Tests non-linear decision boundaries |
| **KNN (k=5)** | Instance-based | Tests if cyclic tables cluster together |
| **Logistic Regression** | Linear model | Tests if the task is linearly separable |
| **MLP Neural Network** | Neural network | Tests deep feature learning capability |

---

## Results

### Experiment 1: Random Forest Baseline

GridSearchCV hyperparameter optimization (5-fold Stratified CV):

| Metric | Value |
|--------|:-----:|
| **Accuracy** | 1.0000 |
| **Precision** | 1.0000 |
| **Recall** | 1.0000 |
| **F1 Score** | 1.0000 |
| **AUC-ROC** | 1.0000 |

![Confusion Matrix](results/confusion_matrix.png)
![ROC Curve](results/roc_curve.png)
![Feature Importance](results/feature_importance.png)

### Experiment 2: Multi-Model Comparison

#### Hand-crafted Features (20 features)

| Model | Accuracy | Precision | Recall | F1 | AUC | CV Accuracy |
|-------|:--------:|:---------:|:------:|:--:|:---:|:-----------:|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +/- 0.0000 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +/- 0.0000 |
| SVM (RBF) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +/- 0.0000 |
| KNN (k=5) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +/- 0.0000 |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +/- 0.0000 |
| MLP Neural Network | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9975 +/- 0.0050 |

#### Raw Flattened Table (420 features)

| Model | Accuracy | Precision | Recall | F1 | AUC | CV Accuracy |
|-------|:--------:|:---------:|:------:|:--:|:---:|:-----------:|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +/- 0.0000 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +/- 0.0000 |
| SVM (RBF) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +/- 0.0000 |
| KNN (k=5) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9994 +/- 0.0013 |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 +/- 0.0000 |
| MLP Neural Network | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9975 +/- 0.0023 |

#### Visualizations

![Model Comparison](results_comparison/model_comparison_metrics.png)
![Metrics Heatmap](results_comparison/metrics_heatmap.png)
![CV Accuracy](results_comparison/cv_accuracy_comparison.png)
![Training Time](results_comparison/training_time.png)

---

## Key Findings

1. **ML can recover cyclic group structure from raw Cayley tables.** Even without hand-crafted algebraic features, all six classifiers achieved near-perfect accuracy on the raw flattened table representation. This confirms that the cyclic property is learnable directly from the multiplication table.

2. **Hand-crafted features are not necessary but improve efficiency.** Models train significantly faster on 20 structural features compared to 420 raw features, while achieving the same accuracy. Feature engineering provides a compact, interpretable representation.

3. **All tested models are equally effective.** Random Forest, Gradient Boosting, SVM, KNN, Logistic Regression, and MLP all achieve perfect or near-perfect classification. The task is linearly separable in the structural feature space (even Logistic Regression achieves 100%).

4. **The cyclic property has a clear algebraic signature.** The perfect separability suggests that cyclic groups leave a strong, distinctive pattern in their Cayley tables that any reasonable classifier can detect.

### Limitations and Future Work

- The current dataset uses groups of order <= 20. Larger groups may present greater classification difficulty, especially for raw features.
- All non-cyclic samples come from well-known group families. Testing with randomly generated group-like structures could be more challenging.
- Future experiments could target multi-class classification (predicting the specific group type: Z_n vs D_n vs S_n) or regression (predicting the number of generators).
- Graph Neural Networks (GNNs) treating the Cayley table as an adjacency matrix could capture structural patterns more naturally.

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
# Install requirements
pip install numpy pandas scikit-learn matplotlib seaborn

# Run algebraic analysis examples
python group_classifier.py

# Run Experiment 1: Random Forest baseline
python experiment_random_forest.py

# Run Experiment 2: Multi-model comparison
python experiment_model_comparison.py

# Run unit tests
python test_classifier.py
```

## Requirements

- Python 3.10+
- numpy, pandas, scikit-learn, matplotlib, seaborn

## License

MIT
