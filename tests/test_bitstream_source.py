"""Tests for TCP bitstream source format decoding."""

import unittest

from sdrparser.audio.input import TCPBitstreamSource


class TestTCPBitstreamSource(unittest.TestCase):
    def test_ascii_bits_format(self):
        src = TCPBitstreamSource(wire_format="ascii-bits")
        bits = src._decode_bytes(b"1010 01\n")
        self.assertEqual(bits, [1, 0, 1, 0, 0, 1])

    def test_dibit_bytes_format(self):
        src = TCPBitstreamSource(wire_format="dibit-bytes")
        bits = src._decode_bytes(bytes([0, 1, 2, 3]))
        self.assertEqual(bits, [0, 0, 0, 1, 1, 0, 1, 1])

    def test_packed_dibits_format(self):
        src = TCPBitstreamSource(wire_format="packed-dibits")
        # 00 01 10 11
        bits = src._decode_bytes(bytes([0b00011011]))
        self.assertEqual(bits, [0, 0, 0, 1, 1, 0, 1, 1])

    def test_auto_detects_ascii_bits(self):
        src = TCPBitstreamSource(wire_format="auto")
        bits = src._decode_bytes(b"0101\n")
        self.assertEqual(bits, [0, 1, 0, 1])

    def test_auto_detects_dibit_bytes(self):
        src = TCPBitstreamSource(wire_format="auto")
        bits = src._decode_bytes(bytes([3, 2, 1, 0]))
        self.assertEqual(bits, [1, 1, 1, 0, 0, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
