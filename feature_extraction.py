"""
Feature Extraction for Cayley Tables
======================================
Cayley tablolarından ML modelleri için öznitelik (feature) çıkarır.
"""

import numpy as np
from collections import Counter


def extract_features(table: np.ndarray) -> dict:
    """
    Bir Cayley tablosundan öznitelik vektörü çıkarır.

    Features:
    ---------
    Yapısal:
        - order: Grup mertebesi (n)
        - is_symmetric: Tablo simetrik mi (abelyen göstergesi)
        - symmetry_ratio: Simetrik giriş oranı

    Eleman mertebeleri:
        - max_element_order: En büyük eleman mertebesi
        - min_element_order: En küçük eleman mertebesi (birim hariç)
        - mean_element_order: Ortalama eleman mertebesi
        - std_element_order: Eleman mertebesi standart sapması
        - max_order_ratio: max_order / n oranı (1.0 ise cyclic)
        - num_generators: Mertebesi n olan eleman sayısı
        - generator_ratio: num_generators / n
        - num_distinct_orders: Farklı mertebe sayısı
        - order_entropy: Mertebe dağılımının entropisi

    İstatistiksel:
        - diagonal_identity_count: Köşegende birim eleman sayısı
        - latin_square: Latin kare mi (her satır/sütun permütasyon)
        - row_uniformity: Satırlardaki değer dağılımı benzerliği
        - trace_value: Köşegen toplamı / n²

    Topolojik:
        - num_involutions: Mertebesi 2 olan eleman sayısı
        - num_subgroup_orders: Farklı alt grup mertebesi sayısı
        - self_inverse_ratio: Kendi tersine eşit olan eleman oranı
    """
    n = len(table)
    features = {}

    # --- Birim eleman ---
    identity = _find_identity(table, n)

    # --- Temel yapısal ---
    features["order"] = n
    features["is_symmetric"] = int(np.array_equal(table, table.T))
    sym_count = sum(1 for i in range(n) for j in range(i + 1, n)
                    if table[i][j] == table[j][i])
    total_pairs = n * (n - 1) / 2 if n > 1 else 1
    features["symmetry_ratio"] = sym_count / total_pairs

    # --- Eleman mertebeleri ---
    orders = []
    for a in range(n):
        orders.append(_element_order(table, a, identity, n))

    features["max_element_order"] = max(orders)
    non_identity_orders = [o for o in orders if o > 1] or [1]
    features["min_element_order"] = min(non_identity_orders)
    features["mean_element_order"] = np.mean(orders)
    features["std_element_order"] = np.std(orders)
    features["max_order_ratio"] = max(orders) / n
    features["num_generators"] = sum(1 for o in orders if o == n)
    features["generator_ratio"] = features["num_generators"] / n
    features["num_distinct_orders"] = len(set(orders))

    # Mertebe dağılımı entropisi
    order_counts = Counter(orders)
    probs = np.array(list(order_counts.values())) / n
    features["order_entropy"] = -np.sum(probs * np.log2(probs + 1e-10))

    # --- Köşegen ---
    diagonal = [table[i][i] for i in range(n)]
    if identity is not None:
        features["diagonal_identity_count"] = sum(1 for d in diagonal if d == identity)
    else:
        features["diagonal_identity_count"] = 0
    features["trace_value"] = sum(diagonal) / (n * n) if n > 0 else 0

    # --- Latin kare ---
    is_latin = True
    for i in range(n):
        if len(set(table[i])) != n or len(set(table[:, i])) != n:
            is_latin = False
            break
    features["latin_square"] = int(is_latin)

    # --- Satır benzerliği ---
    row_counts = []
    for i in range(n):
        counts = Counter(table[i])
        row_counts.append(list(counts.values()))
    avg_std = np.mean([np.std(rc) for rc in row_counts])
    features["row_uniformity"] = avg_std

    # --- İnvolüsyonlar ve tersler ---
    features["num_involutions"] = sum(1 for o in orders if o == 2)

    self_inverse_count = 0
    if identity is not None:
        for a in range(n):
            if table[a][a] == identity:
                self_inverse_count += 1
    features["self_inverse_ratio"] = self_inverse_count / n

    # --- Alt grup mertebeleri (cyclic alt gruplar) ---
    subgroup_orders = set()
    for a in range(n):
        sg = _cyclic_subgroup_order(table, a, identity, n)
        subgroup_orders.add(sg)
    features["num_subgroup_orders"] = len(subgroup_orders)

    # --- Euler phi fonksiyonu oranı ---
    # Cyclic gruplarda generator sayısı = phi(n)
    features["euler_phi_ratio"] = features["num_generators"] / _euler_phi(n) if _euler_phi(n) > 0 else 0

    return features


def extract_features_flat(table: np.ndarray, max_size: int = 30) -> np.ndarray:
    """
    Cayley tablosunu düzleştirilmiş öznitelik vektörüne çevirir.
    Hem yapısal features hem de tablo içeriği birleştirilir.
    """
    struct_features = extract_features(table)
    struct_vec = np.array(list(struct_features.values()), dtype=float)

    # Tablo düzleştirme (sabit boyuta pad)
    n = len(table)
    padded = np.full((max_size, max_size), -1, dtype=float)
    padded[:n, :n] = table / max(n - 1, 1)  # normalize
    flat_table = padded.flatten()

    return np.concatenate([struct_vec, flat_table])


def extract_features_structured(table: np.ndarray) -> np.ndarray:
    """Sadece yapısal öznitelikleri vektör olarak döndürür."""
    features = extract_features(table)
    return np.array(list(features.values()), dtype=float)


def get_feature_names() -> list[str]:
    """Öznitelik isimlerini döndürür."""
    dummy = np.array([[0, 1], [1, 0]])
    features = extract_features(dummy)
    return list(features.keys())


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def _find_identity(table, n):
    for e in range(n):
        if all(table[e][a] == a and table[a][e] == a for a in range(n)):
            return e
    return None


def _element_order(table, a, identity, n):
    if identity is None:
        return -1
    if a == identity:
        return 1
    current = a
    for k in range(1, n + 1):
        if current == identity:
            return k
        current = table[current][a]
    return n + 1


def _cyclic_subgroup_order(table, a, identity, n):
    if identity is None:
        return 1
    visited = {identity}
    current = a
    while current not in visited:
        visited.add(current)
        current = table[current][a]
    return len(visited)


def _euler_phi(n):
    """Euler totient fonksiyonu."""
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


if __name__ == "__main__":
    from dataset_generator import cyclic_group, klein_four_group

    print("=== Z₄ Features ===")
    t = cyclic_group(4)
    feats = extract_features(t)
    for k, v in feats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Klein V₄ Features ===")
    t = klein_four_group()
    feats = extract_features(t)
    for k, v in feats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
