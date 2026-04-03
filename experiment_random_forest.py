"""
Experiment: Random Forest Classification of Cayley Tables
============================================================
Cayley tablolarının cyclic olup olmadığını Random Forest ile sınıflandırır.

Deney adımları:
1. Veri seti üretimi (cyclic ve non-cyclic gruplar)
2. Feature extraction
3. Train/test split
4. Random Forest eğitimi ve hiperparametre optimizasyonu
5. Değerlendirme (accuracy, precision, recall, F1, confusion matrix)
6. Feature importance analizi
7. Sonuçların raporlanması ve görselleştirilmesi
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from dataset_generator import generate_dataset
from feature_extraction import extract_features_structured, get_feature_names


# =============================================================================
# DENEY AYARLARI
# =============================================================================

EXPERIMENT_CONFIG = {
    "name": "Random Forest - Cyclic Group Classification",
    "samples_per_group": 50,
    "max_order": 20,
    "test_size": 0.2,
    "cv_folds": 5,
    "random_seed": 42,
    "output_dir": "results",
}


def run_experiment(config: dict = None) -> dict:
    """Ana deney fonksiyonu."""
    if config is None:
        config = EXPERIMENT_CONFIG

    os.makedirs(config["output_dir"], exist_ok=True)
    results = {"config": config, "timestamp": datetime.now().isoformat()}

    print("=" * 70)
    print("  DENEY: Cayley Table Cyclic/Non-Cyclic Binary Classification")
    print("  Model: Random Forest")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1. VERİ SETİ ÜRETİMİ
    # -----------------------------------------------------------------
    print("\n[1/6] Veri seti üretiliyor...")
    tables, labels, descriptions = generate_dataset(
        samples_per_group=config["samples_per_group"],
        max_order=config["max_order"],
        random_seed=config["random_seed"],
    )

    results["dataset"] = {
        "total_samples": len(tables),
        "cyclic_samples": sum(labels),
        "non_cyclic_samples": len(labels) - sum(labels),
        "unique_group_types": len(set(descriptions)),
    }
    print(f"    Toplam: {len(tables)} örnek")
    print(f"    Cyclic: {sum(labels)}, Non-cyclic: {len(labels) - sum(labels)}")

    # -----------------------------------------------------------------
    # 2. FEATURE EXTRACTION
    # -----------------------------------------------------------------
    print("\n[2/6] Öznitelikler çıkarılıyor...")
    X = np.array([extract_features_structured(t) for t in tables])
    y = np.array(labels)
    feature_names = get_feature_names()

    print(f"    Öznitelik sayısı: {len(feature_names)}")
    print(f"    Feature names: {', '.join(feature_names)}")

    # Feature DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y
    df["group_type"] = descriptions
    df.to_csv(os.path.join(config["output_dir"], "dataset.csv"), index=False)

    results["features"] = {
        "num_features": len(feature_names),
        "feature_names": feature_names,
    }

    # -----------------------------------------------------------------
    # 3. TRAIN/TEST SPLIT
    # -----------------------------------------------------------------
    print("\n[3/6] Train/test ayrımı yapılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_seed"],
        stratify=y,
    )

    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"    Train: {len(X_train)} ({sum(y_train)} cyclic)")
    print(f"    Test:  {len(X_test)} ({sum(y_test)} cyclic)")

    results["split"] = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_cyclic": int(sum(y_train)),
        "test_cyclic": int(sum(y_test)),
    }

    # -----------------------------------------------------------------
    # 4. MODEL EĞİTİMİ + HİPERPARAMETRE OPTİMİZASYONU
    # -----------------------------------------------------------------
    print("\n[4/6] Random Forest eğitiliyor (GridSearch ile)...")
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    rf = RandomForestClassifier(random_state=config["random_seed"])
    cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True,
                         random_state=config["random_seed"])

    grid_search = GridSearchCV(
        rf, param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"    En iyi parametreler: {best_params}")
    print(f"    En iyi CV F1 skoru: {grid_search.best_score_:.4f}")

    results["best_params"] = best_params
    results["best_cv_f1"] = round(grid_search.best_score_, 4)

    # Cross-validation sonuçları
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train,
                                cv=cv, scoring="accuracy")
    print(f"    CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    results["cv_accuracy_mean"] = round(cv_scores.mean(), 4)
    results["cv_accuracy_std"] = round(cv_scores.std(), 4)

    # -----------------------------------------------------------------
    # 5. DEĞERLENDİRME
    # -----------------------------------------------------------------
    print("\n[5/6] Test seti üzerinde değerlendirme...")
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    AUC-ROC:   {auc:.4f}")

    results["test_metrics"] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
    }

    # Classification report
    report = classification_report(y_test, y_pred, target_names=["Non-Cyclic", "Cyclic"])
    print(f"\n    Classification Report:\n{report}")
    results["classification_report"] = classification_report(
        y_test, y_pred, target_names=["Non-Cyclic", "Cyclic"], output_dict=True
    )

    # -----------------------------------------------------------------
    # 6. GÖRSELLEŞTİRME
    # -----------------------------------------------------------------
    print("\n[6/6] Görseller oluşturuluyor...")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Cyclic", "Cyclic"],
                yticklabels=["Non-Cyclic", "Cyclic"], ax=ax)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix — Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "confusion_matrix.png"), dpi=150)
    plt.close()

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"Random Forest (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Random Forest")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "roc_curve.png"), dpi=150)
    plt.close()

    # --- Feature Importance ---
    importances = best_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_k = min(15, len(feature_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(top_k),
        importances[sorted_idx[:top_k]][::-1],
        color="steelblue",
    )
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[:top_k]][::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top Feature Importances — Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "feature_importance.png"), dpi=150)
    plt.close()

    results["feature_importance"] = {
        feature_names[i]: round(importances[i], 4)
        for i in sorted_idx[:top_k]
    }

    # --- Class Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Train/Test dağılımı
    train_counts = [sum(y_train == 0), sum(y_train == 1)]
    test_counts = [sum(y_test == 0), sum(y_test == 1)]
    x_pos = np.arange(2)
    width = 0.35
    axes[0].bar(x_pos - width/2, train_counts, width, label="Train", color="steelblue")
    axes[0].bar(x_pos + width/2, test_counts, width, label="Test", color="coral")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(["Non-Cyclic", "Cyclic"])
    axes[0].set_ylabel("Sayı")
    axes[0].set_title("Sınıf Dağılımı (Train/Test)")
    axes[0].legend()

    # Order dağılımı
    orders = [len(t) for t in tables]
    cyclic_orders = [o for o, l in zip(orders, labels) if l == 1]
    non_cyclic_orders = [o for o, l in zip(orders, labels) if l == 0]
    axes[1].hist(cyclic_orders, bins=20, alpha=0.6, label="Cyclic", color="steelblue")
    axes[1].hist(non_cyclic_orders, bins=20, alpha=0.6, label="Non-Cyclic", color="coral")
    axes[1].set_xlabel("Grup Mertebesi")
    axes[1].set_ylabel("Sayı")
    axes[1].set_title("Grup Mertebesi Dağılımı")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config["output_dir"], "distributions.png"), dpi=150)
    plt.close()

    print(f"    Görseller '{config['output_dir']}/' klasörüne kaydedildi.")

    # -----------------------------------------------------------------
    # SONUÇLARI KAYDET
    # -----------------------------------------------------------------
    with open(os.path.join(config["output_dir"], "experiment_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n    Sonuçlar '{config['output_dir']}/experiment_results.json' dosyasına kaydedildi.")
    print("\n" + "=" * 70)
    print(f"  SONUÇ: Accuracy={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_experiment()
