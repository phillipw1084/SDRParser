"""Tests for dsd-fme backend line parsing."""

import unittest

from sdrparser.backends.dsdfme import parse_dsd_fme_line


class TestDsdfmeLineParser(unittest.TestCase):
    def test_parses_ambe_vocoder_line(self):
        events, proto = parse_dsd_fme_line("DMR AMBE 0123456789ABCD")
        self.assertEqual(proto, "DMR")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "vocoder")
        self.assertEqual(events[0].vocoder_type, "AMBE")
        self.assertEqual(events[0].vocoder_hex, "0123456789ABCD")

    def test_parses_imbe_vocoder_line(self):
        events, proto = parse_dsd_fme_line("P25 IMBE 00112233445566778899AA")
        self.assertEqual(proto, "P25")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "vocoder")
        self.assertEqual(events[0].vocoder_type, "IMBE")

    def test_parses_header_line(self):
        line = "DMR Group Voice Call src=12345 tg=3100 slot=1 color code=1"
        events, proto = parse_dsd_fme_line(line)
        self.assertEqual(proto, "DMR")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "header")
        self.assertEqual(events[0].text, line)

    def test_ignores_non_header_noise(self):
        events, proto = parse_dsd_fme_line("some unrelated timing line")
        self.assertEqual(proto, "Unknown")
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
