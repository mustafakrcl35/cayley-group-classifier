# Cayley Table Group Classifier

Verilen bir Cayley tablosunun geçerli bir grup tanımlayıp tanımlamadığını kontrol eden ve **cyclic olup olmadığını** belirleyen bir Python aracı.

## Özellikler

- **Grup Aksiyomları Kontrolü**: Kapalılık, birim eleman, tersler, birleşme özelliği
- **Cyclic Grup Testi**: Üreteçleri bularak grubun cyclic olup olmadığını belirler
- **Eleman Mertebesi Hesaplama**: Her elemanın mertebesini ve tersini raporlar
- **Alt Grup Keşfi**: Tüm cyclic alt grupları listeler
- **Grup Tanımlama**: Bilinen küçük gruplarla eşleştirme (Z_n, V₄, S₃, D₄, Q₈ vb.)
- **Abelyen Kontrolü**: Değişme özelliğini test eder

## Kullanım

### Hazır Örnekleri Çalıştırma

```bash
python group_classifier.py
```

Bu komut aşağıdaki örnekleri çalıştırır:
- **Z₄** — mod 4 toplama (cyclic)
- **V₄** — Klein dört-grubu (cyclic değil)
- **S₃** — Simetrik grup (non-abelyen, non-cyclic)
- **Z₆** — mod 6 toplama (cyclic)
- **Grup olmayan** bir tablo

### Kendi Tablonuzu Test Etme

```python
from group_classifier import CayleyTableAnalyzer

# Cayley tablosunu 2D liste olarak tanımlayın (0-indexed)
table = [
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
]

analyzer = CayleyTableAnalyzer(table, elements=["0", "1", "2", "3"])
analyzer.full_report()
```

### Tek Tek Kontroller

```python
analyzer = CayleyTableAnalyzer(table)

# Grup mu?
is_grp, details = analyzer.is_group()

# Cyclic mi?
cyclic, msg, generators = analyzer.is_cyclic()

# Abelyen mi?
abelian, msg = analyzer.is_abelian()

# Eleman mertebesi
order = analyzer.element_order(1)
```

## Tablo Formatı

Cayley tablosu `n × n` boyutunda bir 2D listedir. `table[i][j]`, `i * j` işleminin sonucunu verir. Elemanlar `0`'dan `n-1`'e kadar tam sayılarla temsil edilir.

Örnek — Z₃ (mod 3 toplama):

```
  * | 0 | 1 | 2
----|---|---|---
  0 | 0 | 1 | 2
  1 | 1 | 2 | 0
  2 | 2 | 0 | 1
```

```python
table = [
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1],
]
```

## Çıktı Örneği

```
============================================================
     CAYLEY TABLE GRUP ANALİZ RAPORU
============================================================

Grup mertebesi: 4
Elemanlar: {e, a, b, c}

--- Grup Aksiyomları ---
  [✓] Kapalılık
  [✓] Birim eleman: e
  [✓] Tersler
  [✓] Birleşme

--- Cyclic Grup Testi ---
  Cyclic grup DEĞİL. Hiçbir eleman tüm grubu üretmiyor.

--- Grup Tanımlama ---
  ≅ Klein dört-grubu V₄ ≅ Z₂ × Z₂
============================================================
```

## Gereksinimler

- Python 3.10+
- Ek kütüphane gerekmez (sadece standart kütüphane)

## Testler

```bash
python test_classifier.py
```

## Lisans

MIT
