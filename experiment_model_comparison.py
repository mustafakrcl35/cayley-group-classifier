"""
Experiment: Multi-Model Comparison for Cyclic Group Classification
====================================================================
Compares multiple ML classifiers on the cyclic/non-cyclic binary classification
task using two feature representations:
  (A) Hand-crafted structural features (20 features)
  (B) Raw flattened Cayley table (no domain knowledge)

Models tested:
  1. Random Forest
  2. Gradient Boosting (XGBoost-style via HistGradientBoosting)
  3. Support Vector Machine (SVM with RBF kernel)
  4. K-Nearest Neighbors (KNN)
  5. Logistic Regression
  6. Multi-Layer Perceptron (MLP Neural Network)

Research question:
  Can ML recover the element-order structure of a finite group directly
  from its Cayley table, and which models are most effective?
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from dataset_generator import generate_dataset
from feature_extraction import extract_features_structured, extract_features_flat, get_feature_names


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

CONFIG = {
    "name": "Multi-Model Comparison - Cyclic Group Classification",
    "samples_per_group": 50,
    "max_order": 20,
    "max_pad_size": 20,       # Fixed size for raw table flattening
    "test_size": 0.2,
    "cv_folds": 5,
    "random_seed": 42,
    "output_dir": "results_comparison",
}

# Model definitions with default hyperparameters
MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200, max_depth=10, random_state=42
    ),
    "SVM (RBF)": SVC(
        kernel="rbf", C=10, gamma="scale", probability=True, random_state=42
    ),
    "KNN (k=5)": KNeighborsClassifier(
        n_neighbors=5, weights="distance", metric="minkowski"
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, random_state=42
    ),
    "MLP Neural Network": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), max_iter=500,
        early_stopping=True, random_state=42
    ),
}


def run_comparison(config: dict = None) -> dict:
    """Main comparison experiment."""
    if config is None:
        config = CONFIG

    os.makedirs(config["output_dir"], exist_ok=True)
    results = {"config": config, "timestamp": datetime.now().isoformat()}

    print("=" * 75)
    print("  EXPERIMENT: Multi-Model Comparison")
    print("  Task: Cyclic vs Non-Cyclic Binary Classification")
    print("  Feature sets: (A) Hand-crafted  (B) Raw flattened table")
    print("=" * 75)

    # -----------------------------------------------------------------
    # 1. DATASET GENERATION
    # -----------------------------------------------------------------
    print("\n[1/5] Generating dataset...")
    tables, labels, descriptions = generate_dataset(
        samples_per_group=config["samples_per_group"],
        max_order=config["max_order"],
        random_seed=config["random_seed"],
    )
    y = np.array(labels)

    # Filter tables that fit within max_pad_size for raw features
    valid_mask = np.array([len(t) <= config["max_pad_size"] for t in tables])
    tables_filtered = [t for t, v in zip(tables, valid_mask) if v]
    y_filtered = y[valid_mask]
    desc_filtered = [d for d, v in zip(descriptions, valid_mask) if v]

    print(f"    Total samples: {len(tables_filtered)}")
    print(f"    Cyclic: {sum(y_filtered)}, Non-cyclic: {len(y_filtered) - sum(y_filtered)}")
    print(f"    (Filtered to orders <= {config['max_pad_size']})")

    results["dataset"] = {
        "total_samples": len(tables_filtered),
        "cyclic": int(sum(y_filtered)),
        "non_cyclic": int(len(y_filtered) - sum(y_filtered)),
    }

    # -----------------------------------------------------------------
    # 2. FEATURE EXTRACTION
    # -----------------------------------------------------------------
    print("\n[2/5] Extracting features...")

    # (A) Hand-crafted structural features
    X_struct = np.array([extract_features_structured(t) for t in tables_filtered])
    feature_names_struct = get_feature_names()
    print(f"    (A) Hand-crafted: {X_struct.shape[1]} features")

    # (B) Raw flattened Cayley table
    X_raw = np.array([
        extract_features_flat(t, max_size=config["max_pad_size"])
        for t in tables_filtered
    ])
    print(f"    (B) Raw flattened: {X_raw.shape[1]} features "
          f"({len(feature_names_struct)} structural + "
          f"{config['max_pad_size']}x{config['max_pad_size']} table)")

    feature_sets = {
        "Hand-crafted (20 features)": (X_struct, feature_names_struct),
        f"Raw flattened ({X_raw.shape[1]} features)": (X_raw, None),
    }

    # -----------------------------------------------------------------
    # 3. TRAIN/TEST SPLIT
    # -----------------------------------------------------------------
    print("\n[3/5] Splitting data...")
    split_indices = {}
    for fs_name, (X_fs, _) in feature_sets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X_fs, y_filtered,
            test_size=config["test_size"],
            random_state=config["random_seed"],
            stratify=y_filtered,
        )
        split_indices[fs_name] = (X_train, X_test, y_train, y_test)

    n_train = len(split_indices[list(split_indices.keys())[0]][2])
    n_test = len(split_indices[list(split_indices.keys())[0]][3])
    print(f"    Train: {n_train}, Test: {n_test}")

    # -----------------------------------------------------------------
    # 4. MODEL TRAINING & EVALUATION
    # -----------------------------------------------------------------
    print("\n[4/5] Training and evaluating models...")
    print("-" * 75)

    all_results = []
    cv = StratifiedKFold(
        n_splits=config["cv_folds"], shuffle=True,
        random_state=config["random_seed"]
    )

    for fs_name, (X_train, X_test, y_train, y_test) in split_indices.items():
        print(f"\n  Feature set: {fs_name}")
        print(f"  {'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'CV-Acc':>10} {'Time':>8}")
        print(f"  {'-'*80}")

        for model_name, model in MODELS.items():
            # Build pipeline with scaling
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model),
            ])

            # Train
            start_time = time.time()
            pipe.fit(X_train, y_train)
            train_time = time.time() - start_time

            # Predict
            y_pred = pipe.predict(X_test)
            if hasattr(pipe, "predict_proba"):
                try:
                    y_proba = pipe.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                except Exception:
                    auc = -1.0
            else:
                auc = -1.0

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            # Cross-validation
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            print(f"  {model_name:<25} {acc:>7.4f} {prec:>7.4f} {rec:>7.4f} "
                  f"{f1:>7.4f} {auc:>7.4f} {cv_mean:.4f}+/-{cv_std:.4f} {train_time:>7.2f}s")

            all_results.append({
                "feature_set": fs_name,
                "model": model_name,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "auc_roc": round(auc, 4),
                "cv_accuracy_mean": round(cv_mean, 4),
                "cv_accuracy_std": round(cv_std, 4),
                "train_time_seconds": round(train_time, 3),
            })

    results["model_results"] = all_results

    # -----------------------------------------------------------------
    # 5. VISUALIZATION
    # -----------------------------------------------------------------
    print(f"\n[5/5] Generating visualizations...")

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(config["output_dir"], "comparison_results.csv"), index=False)

    # --- Grouped bar chart: Accuracy by model and feature set ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, metric in enumerate(["accuracy", "f1_score"]):
        ax = axes[idx]
        pivot = df_results.pivot(index="model", columns="feature_set", values=metric)
        pivot.plot(kind="barh", ax=ax, color=["steelblue", "coral"])
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} by Model and Feature Set")
        ax.set_xlim(0, 1.05)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "model_comparison_metrics.png"), dpi=150)
    plt.close()

    # --- Heatmap: All metrics ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, fs_name in enumerate(feature_sets.keys()):
        ax = axes[idx]
        subset = df_results[df_results["feature_set"] == fs_name]
        metrics_cols = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
        heatmap_data = subset.set_index("model")[metrics_cols]
        sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu",
                    vmin=0.5, vmax=1.0, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(f"Metrics Heatmap\n{fs_name}")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "metrics_heatmap.png"), dpi=150)
    plt.close()

    # --- Training time comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_time = df_results.pivot(index="model", columns="feature_set", values="train_time_seconds")
    pivot_time.plot(kind="barh", ax=ax, color=["steelblue", "coral"])
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Time by Model and Feature Set")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "training_time.png"), dpi=150)
    plt.close()

    # --- CV Accuracy with error bars ---
    fig, ax = plt.subplots(figsize=(10, 6))
    fs_names = df_results["feature_set"].unique()
    x = np.arange(len(MODELS))
    width = 0.35
    colors = ["steelblue", "coral"]

    for i, fs in enumerate(fs_names):
        subset = df_results[df_results["feature_set"] == fs]
        means = subset["cv_accuracy_mean"].values
        stds = subset["cv_accuracy_std"].values
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds,
                      label=fs, color=colors[i], capsize=4, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(list(MODELS.keys()), rotation=30, ha="right")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("Cross-Validation Accuracy (5-fold) with Std Dev")
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "cv_accuracy_comparison.png"), dpi=150)
    plt.close()

    print(f"    Visualizations saved to '{config['output_dir']}/'.")

    # Save full results
    with open(os.path.join(config["output_dir"], "experiment_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  SUMMARY")
    print("=" * 75)

    for fs_name in feature_sets.keys():
        subset = df_results[df_results["feature_set"] == fs_name]
        best_row = subset.loc[subset["f1_score"].idxmax()]
        print(f"\n  [{fs_name}]")
        print(f"    Best model: {best_row['model']}")
        print(f"    Accuracy={best_row['accuracy']:.4f}  "
              f"F1={best_row['f1_score']:.4f}  "
              f"AUC={best_row['auc_roc']:.4f}")

    # Key finding
    struct_best = df_results[df_results["feature_set"].str.contains("Hand")]["f1_score"].max()
    raw_best = df_results[df_results["feature_set"].str.contains("Raw")]["f1_score"].max()
    print(f"\n  KEY FINDING:")
    print(f"    Hand-crafted features best F1: {struct_best:.4f}")
    print(f"    Raw table features best F1:    {raw_best:.4f}")
    if raw_best >= 0.95:
        print("    -> ML CAN recover group structure from raw Cayley tables!")
    elif raw_best >= 0.80:
        print("    -> ML partially recovers structure; hand-crafted features help significantly.")
    else:
        print("    -> Raw tables alone are insufficient; domain knowledge (features) is critical.")

    print("=" * 75)

    return results


if __name__ == "__main__":
    results = run_comparison()
