"""
Tests for MBE frame interleaving / deinterleaving.
"""

import unittest
from sdrparser.mbe.frames import (
    MBEFrame,
    MBEType,
    deinterleave,
    interleave,
    _DMR_AMBE2_TABLE,
    _NXDN_AMBE2_TABLE,
    _P25_IMBE_TABLE,
    _bits_to_hex,
)


class TestTableLengths(unittest.TestCase):
    """Verify that interleave tables have the correct length."""

    def test_dmr_table_length(self):
        self.assertEqual(len(_DMR_AMBE2_TABLE), 72)

    def test_nxdn_table_length(self):
        self.assertEqual(len(_NXDN_AMBE2_TABLE), 72)

    def test_p25_table_length(self):
        self.assertEqual(len(_P25_IMBE_TABLE), 88)


class TestTablePermutations(unittest.TestCase):
    """Verify that each table is a valid permutation (no duplicates, in-range)."""

    def _check_permutation(self, table, name):
        n = len(table)
        self.assertEqual(sorted(table), list(range(n)),
                         msg=f"{name} is not a valid permutation of 0..{n-1}")

    def test_dmr_is_permutation(self):
        self._check_permutation(_DMR_AMBE2_TABLE, "DMR")

    def test_nxdn_is_permutation(self):
        self._check_permutation(_NXDN_AMBE2_TABLE, "NXDN")

    def test_p25_is_permutation(self):
        self._check_permutation(_P25_IMBE_TABLE, "P25")


class TestRoundTrip(unittest.TestCase):
    """Interleave then deinterleave must recover the original bits."""

    def _roundtrip(self, protocol, n):
        import random
        rng = random.Random(42)
        original = [rng.randint(0, 1) for _ in range(n)]
        table = {
            "DMR":  _DMR_AMBE2_TABLE,
            "NXDN": _NXDN_AMBE2_TABLE,
            "P25":  _P25_IMBE_TABLE,
        }[protocol]
        interleaved   = interleave(original, table)
        deinterleaved = deinterleave(interleaved, table)
        self.assertEqual(deinterleaved, original,
                         msg=f"{protocol} round-trip failed")

    def test_dmr_roundtrip(self):
        self._roundtrip("DMR", 72)

    def test_nxdn_roundtrip(self):
        self._roundtrip("NXDN", 72)

    def test_p25_roundtrip(self):
        self._roundtrip("P25", 88)


class TestMBEFrameFactory(unittest.TestCase):
    """MBEFrame.from_interleaved / from_deinterleaved round-trip."""

    def test_from_interleaved_dmr(self):
        bits = list(range(72))
        bits = [b % 2 for b in bits]
        mf = MBEFrame.from_interleaved("DMR", MBEType.AMBE2, 0, bits)
        self.assertEqual(len(mf.interleaved_bits), 72)
        self.assertEqual(len(mf.deinterleaved_bits), 72)
        # Re-interleaving the deinterleaved bits must give back the original
        reinterleaved = interleave(mf.deinterleaved_bits, _DMR_AMBE2_TABLE)
        self.assertEqual(reinterleaved, bits)

    def test_from_deinterleaved_p25(self):
        bits = [i % 2 for i in range(88)]
        mf = MBEFrame.from_deinterleaved("P25", MBEType.IMBE, 5, bits)
        self.assertEqual(mf.deinterleaved_bits, bits)
        # Deinterleaving the interleaved bits must give back the original
        deinterleaved = deinterleave(mf.interleaved_bits, _P25_IMBE_TABLE)
        self.assertEqual(deinterleaved, bits)

    def test_frame_repr(self):
        bits = [0] * 72
        mf = MBEFrame.from_interleaved("DMR", MBEType.AMBE2, 3, bits)
        r = repr(mf)
        self.assertIn("DMR", r)
        self.assertIn("AMBE2", r)
        self.assertIn("index=3", r)


class TestBitsToHex(unittest.TestCase):
    def test_all_zeros(self):
        self.assertEqual(_bits_to_hex([0] * 8), "00")

    def test_all_ones(self):
        self.assertEqual(_bits_to_hex([1] * 8), "FF")

    def test_known_value(self):
        bits = [0, 1, 0, 1, 0, 1, 0, 1]   # 0x55
        self.assertEqual(_bits_to_hex(bits), "55")

    def test_padding(self):
        # 4 bits → should pad to 8 and represent as one byte
        bits = [1, 0, 1, 0]  # 0xA0 after padding
        result = _bits_to_hex(bits)
        self.assertEqual(result, "A0")

    def test_multi_byte(self):
        bits = [0] * 8 + [1] * 8
        self.assertEqual(_bits_to_hex(bits), "00 FF")


class TestMBEFrameBitStrings(unittest.TestCase):
    def test_bits_str_interleaved(self):
        bits = [1, 0, 1, 0] + [0] * 68
        mf = MBEFrame.from_interleaved("DMR", MBEType.AMBE2, 0, bits)
        s = mf.bits_str("interleaved")
        self.assertTrue(s.startswith("1010"))

    def test_bits_hex_deinterleaved(self):
        bits = [0] * 72
        mf = MBEFrame.from_interleaved("DMR", MBEType.AMBE2, 0, bits)
        h = mf.bits_hex("deinterleaved")
        # All zeros → all "00" bytes
        for tok in h.split():
            self.assertEqual(tok, "00")


if __name__ == "__main__":
    unittest.main()
