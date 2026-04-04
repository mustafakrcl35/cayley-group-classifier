"""
Cayley Table Group Classifier
==============================
Determines whether a given Cayley table defines a valid group,
checks if it is cyclic, and performs detailed group analysis.

Usage:
    python group_classifier.py
"""

from itertools import combinations
from typing import Optional


class CayleyTableAnalyzer:
    """Performs group analysis on a Cayley table."""

    def __init__(self, table: list[list[int]], elements: Optional[list[str]] = None):
        """
        Args:
            table: An n×n Cayley table (with 0-indexed elements).
                   table[i][j] = result of the operation i * j.
            elements: Element names (optional). Defaults to 0, 1, ..., n-1.
        """
        self.table = table
        self.n = len(table)
        self.elements = elements if elements else [str(i) for i in range(self.n)]
        self._identity = None

    def operate(self, a: int, b: int) -> int:
        """Returns a * b from the Cayley table."""
        return self.table[a][b]

    # =========================================================================
    # 1. GROUP AXIOM CHECKS
    # =========================================================================

    def is_closed(self) -> tuple[bool, str]:
        """Closure: Are all results within the set?"""
        for i in range(self.n):
            for j in range(self.n):
                result = self.table[i][j]
                if result < 0 or result >= self.n:
                    return False, (
                        f"Closure violation: {self.elements[i]} * {self.elements[j]} = "
                        f"{result} is outside the set."
                    )
        return True, "Closure is satisfied."

    def find_identity(self) -> tuple[Optional[int], str]:
        """Identity element: Finds e such that e * a = a * e = a."""
        for e in range(self.n):
            is_identity = True
            for a in range(self.n):
                if self.table[e][a] != a or self.table[a][e] != a:
                    is_identity = False
                    break
            if is_identity:
                self._identity = e
                return e, f"Identity element: {self.elements[e]}"
        return None, "Identity element not found!"

    def has_inverses(self) -> tuple[bool, str, dict[int, int]]:
        """Does every element have an inverse? Returns the inverse mapping."""
        if self._identity is None:
            return False, "No identity element, cannot check inverses.", {}

        inverses = {}
        for a in range(self.n):
            found = False
            for b in range(self.n):
                if self.table[a][b] == self._identity and self.table[b][a] == self._identity:
                    inverses[a] = b
                    found = True
                    break
            if not found:
                return False, (
                    f"Inverse of {self.elements[a]} not found!"
                ), inverses
        return True, "Every element has an inverse.", inverses

    def is_associative(self) -> tuple[bool, str]:
        """Associativity: Is (a*b)*c = a*(b*c) for all triples?"""
        for a in range(self.n):
            for b in range(self.n):
                for c in range(self.n):
                    ab = self.table[a][b]
                    bc = self.table[b][c]
                    lhs = self.table[ab][c]
                    rhs = self.table[a][bc]
                    if lhs != rhs:
                        return False, (
                            f"Associativity violation: "
                            f"({self.elements[a]}*{self.elements[b]})*{self.elements[c]} = "
                            f"{self.elements[lhs]}, but "
                            f"{self.elements[a]}*({self.elements[b]}*{self.elements[c]}) = "
                            f"{self.elements[rhs]}"
                        )
        return True, "Associativity is satisfied."

    def is_group(self) -> tuple[bool, list[str]]:
        """Checks all group axioms."""
        results = []

        closed, msg = self.is_closed()
        results.append(f"  [{'✓' if closed else '✗'}] Closure: {msg}")
        if not closed:
            return False, results

        e, msg = self.find_identity()
        results.append(f"  [{'✓' if e is not None else '✗'}] Identity: {msg}")
        if e is None:
            return False, results

        has_inv, msg, _ = self.has_inverses()
        results.append(f"  [{'✓' if has_inv else '✗'}] Inverses: {msg}")

        assoc, msg = self.is_associative()
        results.append(f"  [{'✓' if assoc else '✗'}] Associativity: {msg}")

        is_grp = closed and (e is not None) and has_inv and assoc
        return is_grp, results

    # =========================================================================
    # 2. CYCLIC GROUP ANALYSIS
    # =========================================================================

    def element_order(self, a: int) -> int:
        """Computes the order of an element: the smallest k such that a^k = e."""
        if self._identity is None:
            self.find_identity()
        if self._identity is None:
            return -1

        current = a
        for k in range(1, self.n + 1):
            if current == self._identity:
                return k
            current = self.table[current][a]
        return -1  # Order not found (may occur if not a group)

    def find_generators(self) -> list[int]:
        """Finds cyclic generators: elements whose order equals n."""
        generators = []
        for a in range(self.n):
            if self.element_order(a) == self.n:
                generators.append(a)
        return generators

    def is_cyclic(self) -> tuple[bool, str, list[int]]:
        """Determines whether the group is cyclic."""
        generators = self.find_generators()
        if generators:
            gen_names = [self.elements[g] for g in generators]
            return True, (
                f"CYCLIC GROUP! Generator(s): {', '.join(gen_names)}"
            ), generators
        else:
            return False, "NOT a cyclic group. No single element generates the entire group.", []

    # =========================================================================
    # 3. ADDITIONAL ANALYSES
    # =========================================================================

    def is_abelian(self) -> tuple[bool, str]:
        """Commutativity: Is a*b = b*a for all pairs?"""
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.table[i][j] != self.table[j][i]:
                    return False, (
                        f"Commutativity violation: {self.elements[i]}*{self.elements[j]} != "
                        f"{self.elements[j]}*{self.elements[i]}"
                    )
        return True, "The group is abelian (commutative)."

    def generated_subgroup(self, a: int) -> set[int]:
        """Computes the subgroup <a> generated by element a."""
        if self._identity is None:
            self.find_identity()
        subgroup = set()
        current = self._identity if self._identity is not None else 0
        # Add identity element
        subgroup.add(self._identity)
        current = a
        while current not in subgroup or len(subgroup) == 1:
            subgroup.add(current)
            current = self.table[current][a]
            if current in subgroup:
                break
        return subgroup

    def find_all_subgroups(self) -> list[set[int]]:
        """Finds all subgroups (via cyclic subgroups)."""
        subgroups = []
        seen = []
        for a in range(self.n):
            sg = self.generated_subgroup(a)
            if sg not in seen:
                seen.append(sg)
                subgroups.append(sg)
        return subgroups

    def identify_group(self) -> str:
        """Attempts to match against known small groups."""
        n = self.n
        abelian, _ = self.is_abelian()
        cyclic, _, _ = self.is_cyclic()

        if n == 1:
            return "Trivial group {e}"
        elif n == 2:
            return "Z_2 (cyclic group of order 2)"
        elif n == 3:
            return "Z_3 (cyclic group of order 3)"
        elif n == 4:
            if cyclic:
                return "Z_4 (cyclic group of order 4)"
            else:
                return "Klein four-group V_4 = Z_2 x Z_2"
        elif n == 5:
            return "Z_5 (cyclic group of order 5)"
        elif n == 6:
            if cyclic:
                return "Z_6 (cyclic group of order 6)"
            elif abelian:
                return "Z_2 x Z_3 = Z_6"
            else:
                return "S_3 (symmetric group of degree 3, dihedral group D_3)"
        elif n == 8:
            if cyclic:
                return "Z_8"
            elif abelian:
                # Distinguish Z4xZ2 vs Z2xZ2xZ2
                orders = [self.element_order(a) for a in range(n)]
                if 4 in orders:
                    return "Z_4 x Z_2"
                else:
                    return "Z_2 x Z_2 x Z_2"
            else:
                return "D_4 or Q_8 (dihedral or quaternion)"
        else:
            if cyclic:
                return f"Z_{n} (cyclic group of order {n})"
            elif abelian:
                return f"Abelian group of order {n} (not cyclic)"
            else:
                return f"Non-abelian group of order {n}"

    # =========================================================================
    # 4. REPORTING
    # =========================================================================

    def print_table(self):
        """Pretty-prints the Cayley table."""
        col_w = max(len(e) for e in self.elements) + 1
        header = " * |" + "|".join(e.center(col_w) for e in self.elements)
        separator = "-" * len(header)

        print(separator)
        print(header)
        print(separator)
        for i in range(self.n):
            row = self.elements[i].rjust(col_w - 1) + " |"
            row += "|".join(
                self.elements[self.table[i][j]].center(col_w) for j in range(self.n)
            )
            print(row)
        print(separator)

    def full_report(self):
        """Prints a complete analysis report."""
        print("=" * 60)
        print("     CAYLEY TABLE GROUP ANALYSIS REPORT")
        print("=" * 60)
        print(f"\nGroup order (number of elements): {self.n}")
        print(f"Elements: {{{', '.join(self.elements)}}}\n")

        print("--- Cayley Table ---")
        self.print_table()

        # Group check
        print("\n--- Group Axioms ---")
        is_grp, results = self.is_group()
        for r in results:
            print(r)
        print(f"\n  Result: {'GROUP ✓' if is_grp else 'NOT A GROUP ✗'}")

        if not is_grp:
            print("\n[!] This structure is not a group; further analysis cannot be performed.")
            return

        # Abelian check
        print("\n--- Commutativity ---")
        abelian, msg = self.is_abelian()
        print(f"  {msg}")

        # Element orders
        print("\n--- Element Orders ---")
        _, _, inverses = self.has_inverses()
        for a in range(self.n):
            order = self.element_order(a)
            inv = inverses.get(a, "?")
            inv_name = self.elements[inv] if isinstance(inv, int) else inv
            print(
                f"  |{self.elements[a]}| = {order}, "
                f"inverse: {inv_name}"
            )

        # Cyclic analysis
        print("\n--- Cyclic Group Test ---")
        cyclic, msg, generators = self.is_cyclic()
        print(f"  {msg}")

        if cyclic:
            for g in generators:
                print(f"\n  Elements generated by <{self.elements[g]}>:")
                current = g
                chain = [self.elements[g]]
                for _ in range(self.n - 1):
                    current = self.table[current][g]
                    chain.append(self.elements[current])
                print(f"    {' -> '.join(chain)}")

        # Subgroups
        print("\n--- Subgroups ---")
        subgroups = self.find_all_subgroups()
        for sg in subgroups:
            sg_names = {self.elements[x] for x in sg}
            print(f"  Order {len(sg)}: {{{', '.join(sorted(sg_names))}}}")

        # Group identification
        print("\n--- Group Identification ---")
        identification = self.identify_group()
        print(f"  = {identification}")

        print("\n" + "=" * 60)
        print(f"  RESULT: This structure is a {'CYCLIC' if cyclic else 'NON-CYCLIC'} group.")
        print("=" * 60)


# =============================================================================
# EXAMPLE CAYLEY TABLES
# =============================================================================

def example_z4():
    """Z_4 = {0, 1, 2, 3} under addition mod 4 (cyclic)."""
    print("\n" + "#" * 60)
    print("# EXAMPLE 1: Z_4 (addition mod 4) — Cyclic Group")
    print("#" * 60)
    table = [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
    ]
    analyzer = CayleyTableAnalyzer(table, elements=["0", "1", "2", "3"])
    analyzer.full_report()


def example_klein4():
    """Klein four-group V_4 = {e, a, b, c} — Not cyclic."""
    print("\n" + "#" * 60)
    print("# EXAMPLE 2: Klein Four-Group V_4 — Not Cyclic")
    print("#" * 60)
    table = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]
    analyzer = CayleyTableAnalyzer(table, elements=["e", "a", "b", "c"])
    analyzer.full_report()


def example_s3():
    """S_3 symmetric group (order 6, non-abelian, non-cyclic)."""
    print("\n" + "#" * 60)
    print("# EXAMPLE 3: S_3 Symmetric Group — Non-Abelian, Non-Cyclic")
    print("#" * 60)
    # S_3 elements: e, r, r^2, s, sr, sr^2
    # (r = 120° rotation, s = reflection)
    table = [
        [0, 1, 2, 3, 4, 5],
        [1, 2, 0, 4, 5, 3],
        [2, 0, 1, 5, 3, 4],
        [3, 5, 4, 0, 2, 1],
        [4, 3, 5, 1, 0, 2],
        [5, 4, 3, 2, 1, 0],
    ]
    elements = ["e", "r", "r^2", "s", "sr", "sr^2"]
    analyzer = CayleyTableAnalyzer(table, elements=elements)
    analyzer.full_report()


def example_z6():
    """Z_6 = {0,1,2,3,4,5} under addition mod 6 (cyclic)."""
    print("\n" + "#" * 60)
    print("# EXAMPLE 4: Z_6 (addition mod 6) — Cyclic Group")
    print("#" * 60)
    table = [[(i + j) % 6 for j in range(6)] for i in range(6)]
    analyzer = CayleyTableAnalyzer(table)
    analyzer.full_report()


def example_not_a_group():
    """Example of a table that is not a group."""
    print("\n" + "#" * 60)
    print("# EXAMPLE 5: Not a Group")
    print("#" * 60)
    table = [
        [0, 1, 2],
        [1, 1, 2],
        [2, 2, 0],
    ]
    analyzer = CayleyTableAnalyzer(table, elements=["a", "b", "c"])
    analyzer.full_report()


# =============================================================================
# CUSTOM TABLE INPUT
# =============================================================================

def custom_table():
    """Enter your own Cayley table here."""
    print("\n" + "#" * 60)
    print("# CUSTOM TABLE: Test your own table")
    print("#" * 60)

    # ---- ENTER YOUR TABLE HERE ----
    # Number elements starting from 0.
    # table[i][j] = result of i * j

    table = [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
        [3, 4, 0, 1, 2],
        [4, 0, 1, 2, 3],
    ]
    elements = ["0", "1", "2", "3", "4"]  # Optional custom names

    analyzer = CayleyTableAnalyzer(table, elements=elements)
    analyzer.full_report()


if __name__ == "__main__":
    example_z4()
    example_klein4()
    example_s3()
    example_z6()
    example_not_a_group()

    # To test your own table:
    # custom_table()
