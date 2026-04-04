"""
Cayley Table Group Classifier - Test Suite
"""

import unittest
from group_classifier import CayleyTableAnalyzer


class TestGroupAxioms(unittest.TestCase):
    """Tests for group axiom verification."""

    def test_z4_is_group(self):
        table = [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
        ]
        a = CayleyTableAnalyzer(table)
        is_grp, _ = a.is_group()
        self.assertTrue(is_grp)

    def test_invalid_table_not_group(self):
        table = [
            [0, 1, 2],
            [1, 1, 2],
            [2, 2, 0],
        ]
        a = CayleyTableAnalyzer(table)
        is_grp, _ = a.is_group()
        self.assertFalse(is_grp)

    def test_identity_z4(self):
        table = [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
        ]
        a = CayleyTableAnalyzer(table)
        e, _ = a.find_identity()
        self.assertEqual(e, 0)

    def test_s3_is_group(self):
        table = [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 0, 4, 5, 3],
            [2, 0, 1, 5, 3, 4],
            [3, 5, 4, 0, 2, 1],
            [4, 3, 5, 1, 0, 2],
            [5, 4, 3, 2, 1, 0],
        ]
        a = CayleyTableAnalyzer(table)
        is_grp, _ = a.is_group()
        self.assertTrue(is_grp)


class TestCyclicDetection(unittest.TestCase):
    """Tests for cyclic group detection."""

    def test_z4_is_cyclic(self):
        table = [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
        ]
        a = CayleyTableAnalyzer(table)
        a.is_group()
        cyclic, _, gens = a.is_cyclic()
        self.assertTrue(cyclic)
        self.assertIn(1, gens)
        self.assertIn(3, gens)

    def test_klein4_not_cyclic(self):
        table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
        ]
        a = CayleyTableAnalyzer(table)
        a.is_group()
        cyclic, _, _ = a.is_cyclic()
        self.assertFalse(cyclic)

    def test_z6_is_cyclic(self):
        table = [[(i + j) % 6 for j in range(6)] for i in range(6)]
        a = CayleyTableAnalyzer(table)
        a.is_group()
        cyclic, _, gens = a.is_cyclic()
        self.assertTrue(cyclic)
        self.assertIn(1, gens)
        self.assertIn(5, gens)

    def test_s3_not_cyclic(self):
        table = [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 0, 4, 5, 3],
            [2, 0, 1, 5, 3, 4],
            [3, 5, 4, 0, 2, 1],
            [4, 3, 5, 1, 0, 2],
            [5, 4, 3, 2, 1, 0],
        ]
        a = CayleyTableAnalyzer(table)
        a.is_group()
        cyclic, _, _ = a.is_cyclic()
        self.assertFalse(cyclic)

    def test_z5_is_cyclic(self):
        table = [[(i + j) % 5 for j in range(5)] for i in range(5)]
        a = CayleyTableAnalyzer(table)
        a.is_group()
        cyclic, _, _ = a.is_cyclic()
        self.assertTrue(cyclic)

    def test_z2_is_cyclic(self):
        table = [[0, 1], [1, 0]]
        a = CayleyTableAnalyzer(table)
        a.is_group()
        cyclic, _, _ = a.is_cyclic()
        self.assertTrue(cyclic)


class TestElementOrders(unittest.TestCase):
    """Tests for element order computation."""

    def test_z4_orders(self):
        table = [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
        ]
        a = CayleyTableAnalyzer(table)
        a.find_identity()
        self.assertEqual(a.element_order(0), 1)  # identity
        self.assertEqual(a.element_order(1), 4)
        self.assertEqual(a.element_order(2), 2)
        self.assertEqual(a.element_order(3), 4)

    def test_klein4_orders(self):
        table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
        ]
        a = CayleyTableAnalyzer(table)
        a.find_identity()
        self.assertEqual(a.element_order(0), 1)
        self.assertEqual(a.element_order(1), 2)
        self.assertEqual(a.element_order(2), 2)
        self.assertEqual(a.element_order(3), 2)


class TestAbelian(unittest.TestCase):
    """Tests for abelian group detection."""

    def test_z4_is_abelian(self):
        table = [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
        ]
        a = CayleyTableAnalyzer(table)
        abelian, _ = a.is_abelian()
        self.assertTrue(abelian)

    def test_s3_not_abelian(self):
        table = [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 0, 4, 5, 3],
            [2, 0, 1, 5, 3, 4],
            [3, 5, 4, 0, 2, 1],
            [4, 3, 5, 1, 0, 2],
            [5, 4, 3, 2, 1, 0],
        ]
        a = CayleyTableAnalyzer(table)
        abelian, _ = a.is_abelian()
        self.assertFalse(abelian)


class TestGroupIdentification(unittest.TestCase):
    """Tests for group identification."""

    def test_identify_z4(self):
        table = [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
        ]
        a = CayleyTableAnalyzer(table)
        a.is_group()
        name = a.identify_group()
        self.assertIn("Z_4", name)

    def test_identify_klein4(self):
        table = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
        ]
        a = CayleyTableAnalyzer(table)
        a.is_group()
        name = a.identify_group()
        self.assertIn("Klein", name)

    def test_identify_s3(self):
        table = [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 0, 4, 5, 3],
            [2, 0, 1, 5, 3, 4],
            [3, 5, 4, 0, 2, 1],
            [4, 3, 5, 1, 0, 2],
            [5, 4, 3, 2, 1, 0],
        ]
        a = CayleyTableAnalyzer(table)
        a.is_group()
        name = a.identify_group()
        self.assertIn("S_3", name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
