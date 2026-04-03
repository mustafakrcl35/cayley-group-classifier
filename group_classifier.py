"""
Cayley Table Group Classifier
==============================
Verilen bir Cayley tablosunun geçerli bir grup olup olmadığını,
cyclic olup olmadığını ve detaylı grup analizini yapar.

Kullanım:
    python group_classifier.py
"""

from itertools import combinations
from typing import Optional


class CayleyTableAnalyzer:
    """Cayley tablosu üzerinde grup analizi yapan sınıf."""

    def __init__(self, table: list[list[int]], elements: Optional[list[str]] = None):
        """
        Args:
            table: n×n boyutunda Cayley tablosu (0-indexed elemanlarla).
                   table[i][j] = i * j işleminin sonucu.
            elements: Eleman isimleri (opsiyonel). Verilmezse 0,1,...,n-1 kullanılır.
        """
        self.table = table
        self.n = len(table)
        self.elements = elements if elements else [str(i) for i in range(self.n)]
        self._identity = None

    def operate(self, a: int, b: int) -> int:
        """a * b işlemini Cayley tablosundan döndürür."""
        return self.table[a][b]

    # =========================================================================
    # 1. GRUP AKSIYOMLARI KONTROLLERI
    # =========================================================================

    def is_closed(self) -> tuple[bool, str]:
        """Kapalılık: Tüm sonuçlar küme içinde mi?"""
        for i in range(self.n):
            for j in range(self.n):
                result = self.table[i][j]
                if result < 0 or result >= self.n:
                    return False, (
                        f"Kapalılık ihlali: {self.elements[i]} * {self.elements[j]} = "
                        f"{result} küme dışında."
                    )
        return True, "Kapalılık sağlanıyor."

    def find_identity(self) -> tuple[Optional[int], str]:
        """Birim eleman: e * a = a * e = a olan e'yi bulur."""
        for e in range(self.n):
            is_identity = True
            for a in range(self.n):
                if self.table[e][a] != a or self.table[a][e] != a:
                    is_identity = False
                    break
            if is_identity:
                self._identity = e
                return e, f"Birim eleman: {self.elements[e]}"
        return None, "Birim eleman bulunamadı!"

    def has_inverses(self) -> tuple[bool, str, dict[int, int]]:
        """Her elemanın tersi var mı? Ters eşlemesini döndürür."""
        if self._identity is None:
            return False, "Birim eleman yok, ters kontrol edilemez.", {}

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
                    f"{self.elements[a]} elemanının tersi bulunamadı!"
                ), inverses
        return True, "Her elemanın tersi mevcut.", inverses

    def is_associative(self) -> tuple[bool, str]:
        """Birleşme özelliği: (a*b)*c = a*(b*c) tüm üçlüler için sağlanıyor mu?"""
        for a in range(self.n):
            for b in range(self.n):
                for c in range(self.n):
                    ab = self.table[a][b]
                    bc = self.table[b][c]
                    lhs = self.table[ab][c]
                    rhs = self.table[a][bc]
                    if lhs != rhs:
                        return False, (
                            f"Birleşme ihlali: "
                            f"({self.elements[a]}*{self.elements[b]})*{self.elements[c]} = "
                            f"{self.elements[lhs]}, ama "
                            f"{self.elements[a]}*({self.elements[b]}*{self.elements[c]}) = "
                            f"{self.elements[rhs]}"
                        )
        return True, "Birleşme özelliği sağlanıyor."

    def is_group(self) -> tuple[bool, list[str]]:
        """Tüm grup aksiyomlarını kontrol eder."""
        results = []

        closed, msg = self.is_closed()
        results.append(f"  [{'✓' if closed else '✗'}] Kapalılık: {msg}")
        if not closed:
            return False, results

        e, msg = self.find_identity()
        results.append(f"  [{'✓' if e is not None else '✗'}] Birim eleman: {msg}")
        if e is None:
            return False, results

        has_inv, msg, _ = self.has_inverses()
        results.append(f"  [{'✓' if has_inv else '✗'}] Tersler: {msg}")

        assoc, msg = self.is_associative()
        results.append(f"  [{'✓' if assoc else '✗'}] Birleşme: {msg}")

        is_grp = closed and (e is not None) and has_inv and assoc
        return is_grp, results

    # =========================================================================
    # 2. CYCLIC GRUP ANALİZİ
    # =========================================================================

    def element_order(self, a: int) -> int:
        """Bir elemanın mertebesini hesaplar: a^k = e olan en küçük k."""
        if self._identity is None:
            self.find_identity()
        if self._identity is None:
            return -1

        current = a
        for k in range(1, self.n + 1):
            if current == self._identity:
                return k
            current = self.table[current][a]
        return -1  # Mertebe bulunamadı (grup değilse olabilir)

    def find_generators(self) -> list[int]:
        """Cyclic üreteçleri bulur: mertebesi n olan elemanlar."""
        generators = []
        for a in range(self.n):
            if self.element_order(a) == self.n:
                generators.append(a)
        return generators

    def is_cyclic(self) -> tuple[bool, str, list[int]]:
        """Grubun cyclic olup olmadığını belirler."""
        generators = self.find_generators()
        if generators:
            gen_names = [self.elements[g] for g in generators]
            return True, (
                f"CYCLIC GRUP! Üreteç(ler): {', '.join(gen_names)}"
            ), generators
        else:
            return False, "Cyclic grup DEĞİL. Hiçbir eleman tüm grubu üretmiyor.", []

    # =========================================================================
    # 3. EK ANALİZLER
    # =========================================================================

    def is_abelian(self) -> tuple[bool, str]:
        """Değişme özelliği: a*b = b*a tüm çiftler için?"""
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.table[i][j] != self.table[j][i]:
                    return False, (
                        f"Değişme ihlali: {self.elements[i]}*{self.elements[j]} ≠ "
                        f"{self.elements[j]}*{self.elements[i]}"
                    )
        return True, "Grup abelyen (değişmeli)."

    def generated_subgroup(self, a: int) -> set[int]:
        """<a> alt grubunu hesaplar."""
        if self._identity is None:
            self.find_identity()
        subgroup = set()
        current = self._identity if self._identity is not None else 0
        # Birim elemanı ekle
        subgroup.add(self._identity)
        current = a
        while current not in subgroup or len(subgroup) == 1:
            subgroup.add(current)
            current = self.table[current][a]
            if current in subgroup:
                break
        return subgroup

    def find_all_subgroups(self) -> list[set[int]]:
        """Tüm alt grupları bulur (cyclic alt gruplar üzerinden)."""
        subgroups = []
        seen = []
        for a in range(self.n):
            sg = self.generated_subgroup(a)
            if sg not in seen:
                seen.append(sg)
                subgroups.append(sg)
        return subgroups

    def identify_group(self) -> str:
        """Bilinen küçük gruplarla eşleştirmeye çalışır."""
        n = self.n
        abelian, _ = self.is_abelian()
        cyclic, _, _ = self.is_cyclic()

        if n == 1:
            return "Trivial grup {e}"
        elif n == 2:
            return "Z₂ (2. mertebeden cyclic grup)"
        elif n == 3:
            return "Z₃ (3. mertebeden cyclic grup)"
        elif n == 4:
            if cyclic:
                return "Z₄ (4. mertebeden cyclic grup)"
            else:
                return "Klein dört-grubu V₄ ≅ Z₂ × Z₂"
        elif n == 5:
            return "Z₅ (5. mertebeden cyclic grup)"
        elif n == 6:
            if cyclic:
                return "Z₆ (6. mertebeden cyclic grup)"
            elif abelian:
                return "Z₂ × Z₃ ≅ Z₆"
            else:
                return "S₃ (3. dereceden simetrik grup, dihedral grup D₃)"
        elif n == 8:
            if cyclic:
                return "Z₈"
            elif abelian:
                # Z4×Z2 vs Z2×Z2×Z2 ayrımı
                orders = [self.element_order(a) for a in range(n)]
                if 4 in orders:
                    return "Z₄ × Z₂"
                else:
                    return "Z₂ × Z₂ × Z₂"
            else:
                return "D₄ veya Q₈ (dihedral veya kuaterniyon)"
        else:
            if cyclic:
                return f"Z_{n} ({n}. mertebeden cyclic grup)"
            elif abelian:
                return f"{n}. mertebeden abelyen grup (cyclic değil)"
            else:
                return f"{n}. mertebeden non-abelyen grup"

    # =========================================================================
    # 4. RAPORLAMA
    # =========================================================================

    def print_table(self):
        """Cayley tablosunu güzel biçimde yazdırır."""
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
        """Tam analiz raporu yazdırır."""
        print("=" * 60)
        print("     CAYLEY TABLE GRUP ANALİZ RAPORU")
        print("=" * 60)
        print(f"\nGrup mertebesi (eleman sayısı): {self.n}")
        print(f"Elemanlar: {{{', '.join(self.elements)}}}\n")

        print("--- Cayley Tablosu ---")
        self.print_table()

        # Grup kontrolü
        print("\n--- Grup Aksiyomları ---")
        is_grp, results = self.is_group()
        for r in results:
            print(r)
        print(f"\n  Sonuç: {'GRUP ✓' if is_grp else 'GRUP DEĞİL ✗'}")

        if not is_grp:
            print("\n[!] Bu yapı bir grup olmadığından ileri analiz yapılamıyor.")
            return

        # Abelyen kontrolü
        print("\n--- Değişme Özelliği ---")
        abelian, msg = self.is_abelian()
        print(f"  {msg}")

        # Eleman mertebeleri
        print("\n--- Eleman Mertebeleri ---")
        _, _, inverses = self.has_inverses()
        for a in range(self.n):
            order = self.element_order(a)
            inv = inverses.get(a, "?")
            inv_name = self.elements[inv] if isinstance(inv, int) else inv
            print(
                f"  |{self.elements[a]}| = {order}, "
                f"tersi: {inv_name}"
            )

        # Cyclic analizi
        print("\n--- Cyclic Grup Testi ---")
        cyclic, msg, generators = self.is_cyclic()
        print(f"  {msg}")

        if cyclic:
            for g in generators:
                print(f"\n  Üreteç <{self.elements[g]}> ile üretilen elemanlar:")
                current = g
                chain = [self.elements[g]]
                for _ in range(self.n - 1):
                    current = self.table[current][g]
                    chain.append(self.elements[current])
                print(f"    {' → '.join(chain)}")

        # Alt gruplar
        print("\n--- Alt Gruplar ---")
        subgroups = self.find_all_subgroups()
        for sg in subgroups:
            sg_names = {self.elements[x] for x in sg}
            print(f"  Mertebe {len(sg)}: {{{', '.join(sorted(sg_names))}}}")

        # Grup tanımlama
        print("\n--- Grup Tanımlama ---")
        identification = self.identify_group()
        print(f"  ≅ {identification}")

        print("\n" + "=" * 60)
        print(f"  SONUÇ: Bu yapı {'CYCLIC' if cyclic else 'CYCLIC OLMAYAN'} bir gruptur.")
        print("=" * 60)


# =============================================================================
# ÖRNEK CAYLEY TABLOLARI
# =============================================================================

def example_z4():
    """Z₄ = {0, 1, 2, 3} mod 4 toplama altında (cyclic)."""
    print("\n" + "#" * 60)
    print("# ÖRNEK 1: Z₄ (mod 4 toplama) — Cyclic Grup")
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
    """Klein dört-grubu V₄ = {e, a, b, c} — Cyclic değil."""
    print("\n" + "#" * 60)
    print("# ÖRNEK 2: Klein Dört-Grubu V₄ — Cyclic Değil")
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
    """S₃ simetrik grup (mertebe 6, non-abelyen, non-cyclic)."""
    print("\n" + "#" * 60)
    print("# ÖRNEK 3: S₃ Simetrik Grup — Non-Abelyen, Non-Cyclic")
    print("#" * 60)
    # S₃ elemanları: e, r, r², s, sr, sr²
    # (r = 120° dönme, s = yansıma)
    table = [
        [0, 1, 2, 3, 4, 5],
        [1, 2, 0, 4, 5, 3],
        [2, 0, 1, 5, 3, 4],
        [3, 5, 4, 0, 2, 1],
        [4, 3, 5, 1, 0, 2],
        [5, 4, 3, 2, 1, 0],
    ]
    elements = ["e", "r", "r²", "s", "sr", "sr²"]
    analyzer = CayleyTableAnalyzer(table, elements=elements)
    analyzer.full_report()


def example_z6():
    """Z₆ = {0,1,2,3,4,5} mod 6 toplama (cyclic)."""
    print("\n" + "#" * 60)
    print("# ÖRNEK 4: Z₆ (mod 6 toplama) — Cyclic Grup")
    print("#" * 60)
    table = [[(i + j) % 6 for j in range(6)] for i in range(6)]
    analyzer = CayleyTableAnalyzer(table)
    analyzer.full_report()


def example_not_a_group():
    """Grup olmayan bir tablo örneği."""
    print("\n" + "#" * 60)
    print("# ÖRNEK 5: Grup Olmayan Tablo")
    print("#" * 60)
    table = [
        [0, 1, 2],
        [1, 1, 2],
        [2, 2, 0],
    ]
    analyzer = CayleyTableAnalyzer(table, elements=["a", "b", "c"])
    analyzer.full_report()


# =============================================================================
# KULLANICI GİRİŞİ
# =============================================================================

def custom_table():
    """Kendi Cayley tablonuzu buraya girin."""
    print("\n" + "#" * 60)
    print("# ÖZEL TABLO: Kendi tablonuzu test edin")
    print("#" * 60)

    # ---- BURAYA KENDİ TABLONUZU GİRİN ----
    # Elemanları 0'dan başlayarak numaralandırın.
    # table[i][j] = i işlem j'nin sonucu

    table = [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
        [3, 4, 0, 1, 2],
        [4, 0, 1, 2, 3],
    ]
    elements = ["0", "1", "2", "3", "4"]  # İsteğe bağlı isimler

    analyzer = CayleyTableAnalyzer(table, elements=elements)
    analyzer.full_report()


if __name__ == "__main__":
    example_z4()
    example_klein4()
    example_s3()
    example_z6()
    example_not_a_group()

    # Kendi tablonuzu test etmek için:
    # custom_table()
