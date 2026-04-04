"""
Cayley Table Dataset Generator
================================
Generates cyclic and non-cyclic Cayley tables from known group structures.
Creates a labeled dataset for ML experiments.
"""

import numpy as np
import random
from itertools import product as iter_product


# =============================================================================
# GROUP GENERATORS
# =============================================================================

def cyclic_group(n: int) -> np.ndarray:
    """Generates the Cayley table for the cyclic group Z_n (addition mod n)."""
    return np.array([[(i + j) % n for j in range(n)] for i in range(n)])


def dihedral_group(n: int) -> np.ndarray:
    """
    Generates the Cayley table for the dihedral group D_n.
    Order: 2n. Elements: r^0, r^1, ..., r^(n-1), s, sr, ..., sr^(n-1)
    Requires n >= 3.
    """
    order = 2 * n
    table = np.zeros((order, order), dtype=int)

    for i in range(order):
        for j in range(order):
            # Convert i and j to (flip, rotation) pairs
            fi, ri = divmod(i, n)  # fi: 0 or 1, ri: 0..n-1
            fj, rj = divmod(j, n)

            if fi == 0:
                # r^ri * (s^fj * r^rj) = s^fj * r^(ri+rj or -ri+rj)
                if fj == 0:
                    new_f = 0
                    new_r = (ri + rj) % n
                else:
                    new_f = 1
                    new_r = (-ri + rj) % n
            else:
                # s*r^ri * (s^fj * r^rj)
                if fj == 0:
                    new_f = 1
                    new_r = (ri + rj) % n
                else:
                    new_f = 0
                    new_r = (-ri + rj) % n

            table[i][j] = new_f * n + new_r

    return table


def direct_product(table_a: np.ndarray, table_b: np.ndarray) -> np.ndarray:
    """Generates the Cayley table for the direct product of two groups."""
    na = len(table_a)
    nb = len(table_b)
    n = na * nb
    table = np.zeros((n, n), dtype=int)

    for i in range(n):
        ai, bi = divmod(i, nb)
        for j in range(n):
            aj, bj = divmod(j, nb)
            result_a = table_a[ai][aj]
            result_b = table_b[bi][bj]
            table[i][j] = result_a * nb + result_b

    return table


def klein_four_group() -> np.ndarray:
    """Klein four-group V_4 = Z_2 x Z_2."""
    return direct_product(cyclic_group(2), cyclic_group(2))


def symmetric_group_s3() -> np.ndarray:
    """S_3 symmetric group (order 6)."""
    # Permutations: e, (123), (132), (12), (13), (23)
    perms = [
        (0, 1, 2), (1, 2, 0), (2, 0, 1),
        (1, 0, 2), (2, 1, 0), (0, 2, 1)
    ]

    def compose(p, q):
        return tuple(p[q[i]] for i in range(3))

    n = len(perms)
    table = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            result = compose(perms[i], perms[j])
            table[i][j] = perms.index(result)
    return table


def quaternion_group() -> np.ndarray:
    """Q_8 quaternion group (order 8)."""
    # Elements: 1, -1, i, -i, j, -j, k, -k (indexed 0-7)
    table = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 3, 2, 5, 4, 7, 6],
        [2, 3, 1, 0, 6, 7, 5, 4],
        [3, 2, 0, 1, 7, 6, 4, 5],
        [4, 5, 7, 6, 1, 0, 2, 3],
        [5, 4, 6, 7, 0, 1, 3, 2],
        [6, 7, 4, 5, 3, 2, 1, 0],
        [7, 6, 5, 4, 2, 3, 0, 1],
    ])
    return table


def random_permutation_relabeling(table: np.ndarray) -> np.ndarray:
    """
    Randomly relabels the elements in a Cayley table.
    This produces isomorphic but visually distinct tables.
    """
    n = len(table)
    perm = list(range(n))
    random.shuffle(perm)
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i

    new_table = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            new_table[perm[i]][perm[j]] = perm[table[i][j]]
    return new_table


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_dataset(
    samples_per_group: int = 50,
    max_order: int = 30,
    random_seed: int = 42,
    fixed_size: int = None,
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """
    Generates a dataset of Cayley tables from cyclic and non-cyclic groups.

    Args:
        samples_per_group: Number of permutation relabelings per group structure.
        max_order: Maximum order for generated cyclic groups.
        random_seed: Random seed for reproducibility.
        fixed_size: If provided, all tables are padded to this size.

    Returns:
        tables: List of Cayley tables
        labels: 1 = cyclic, 0 = non-cyclic
        descriptions: Group type descriptions
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    tables = []
    labels = []
    descriptions = []

    # --- CYCLIC GROUPS ---
    for n in range(2, max_order + 1):
        base_table = cyclic_group(n)
        for _ in range(samples_per_group):
            t = random_permutation_relabeling(base_table)
            tables.append(t)
            labels.append(1)
            descriptions.append(f"Z_{n}")

    # --- NON-CYCLIC GROUPS ---

    # Klein four-group V_4 (order 4)
    base = klein_four_group()
    for _ in range(samples_per_group):
        tables.append(random_permutation_relabeling(base))
        labels.append(0)
        descriptions.append("V_4")

    # S_3 (order 6)
    base = symmetric_group_s3()
    for _ in range(samples_per_group):
        tables.append(random_permutation_relabeling(base))
        labels.append(0)
        descriptions.append("S_3")

    # Dihedral groups D_n (order 2n, n >= 3)
    for n in range(3, max_order // 2 + 1):
        base = dihedral_group(n)
        for _ in range(samples_per_group):
            tables.append(random_permutation_relabeling(base))
            labels.append(0)
            descriptions.append(f"D_{n}")

    # Q_8 quaternion group (order 8)
    base = quaternion_group()
    for _ in range(samples_per_group):
        tables.append(random_permutation_relabeling(base))
        labels.append(0)
        descriptions.append("Q_8")

    # Direct products (non-cyclic ones: gcd(m, n) > 1)
    non_cyclic_products = [
        (2, 2), (2, 4), (2, 6), (2, 8), (2, 10),
        (3, 3), (3, 6), (3, 9),
        (4, 4), (4, 2), (2, 2, 2),
    ]
    for factors in non_cyclic_products:
        if len(factors) == 2:
            m, k = factors
            base = direct_product(cyclic_group(m), cyclic_group(k))
            name = f"Z_{m}xZ_{k}"
        else:
            m, k, l = factors
            base = direct_product(
                direct_product(cyclic_group(m), cyclic_group(k)),
                cyclic_group(l)
            )
            name = f"Z_{m}xZ_{k}xZ_{l}"

        # Only include groups that are actually non-cyclic
        order = len(base)
        is_actually_cyclic = _check_cyclic(base)
        if not is_actually_cyclic:
            for _ in range(samples_per_group):
                tables.append(random_permutation_relabeling(base))
                labels.append(0)
                descriptions.append(name)

    # Fixed-size padding (optional)
    if fixed_size is not None:
        tables = [_pad_table(t, fixed_size) for t in tables]

    return tables, labels, descriptions


def _check_cyclic(table: np.ndarray) -> bool:
    """Checks whether a Cayley table represents a cyclic group."""
    n = len(table)
    # Find identity element
    identity = None
    for e in range(n):
        if all(table[e][a] == a and table[a][e] == a for a in range(n)):
            identity = e
            break
    if identity is None:
        return False

    # Is there an element of order n?
    for a in range(n):
        current = a
        order = 1
        while current != identity and order <= n:
            current = table[current][a]
            order += 1
        if current == identity and order == n:
            return True
    return False


def _pad_table(table: np.ndarray, size: int) -> np.ndarray:
    """Pads the table to the given size (padding with -1)."""
    n = len(table)
    if n >= size:
        return table[:size, :size]
    padded = np.full((size, size), -1, dtype=int)
    padded[:n, :n] = table
    return padded


if __name__ == "__main__":
    tables, labels, descs = generate_dataset(samples_per_group=10, max_order=10)
    print(f"Total samples: {len(tables)}")
    print(f"Cyclic: {sum(labels)}, Non-cyclic: {len(labels) - sum(labels)}")
    print(f"Unique group types: {len(set(descs))}")

    # Distribution by group type
    from collections import Counter
    dist = Counter(descs)
    for group, count in sorted(dist.items()):
        label = labels[descs.index(group)]
        tag = "cyclic" if label == 1 else "non-cyclic"
        print(f"  {group}: {count} samples ({tag})")
