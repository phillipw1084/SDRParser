"""
Tests for the P25 Phase 1 decoder.
"""

import unittest
from sdrparser.protocols.p25 import (
    P25Decoder,
    P25_SYNC,
    P25_SYNC_LEN,
    P25_NID_LEN,
    DUID_HDU,
    DUID_LDU1,
    DUID_TDU,
    _parse_nid,
    _parse_hdu,
    _parse_ldu1_lc,
    _extract_imbe_from_ldu,
)
from sdrparser.protocols.base import FrameKind
from sdrparser.dsp.demod import int_to_bits


def _nid_bits(nac: int, duid: int) -> list:
    """Build a 64-bit NID field (NAC + DUID + zeros for RS parity)."""
    bits = int_to_bits(nac, 12) + int_to_bits(duid, 4) + [0] * 48
    return bits


def _make_p25_frame(duid: int, payload_bits: list | None = None,
                    nac: int = 0x293) -> list:
    """Build a synthetic P25 frame: sync + NID + optional payload."""
    frame = list(P25_SYNC) + _nid_bits(nac, duid)
    if payload_bits:
        frame.extend(payload_bits)
    return frame


class TestSyncPattern(unittest.TestCase):
    def test_sync_length(self):
        self.assertEqual(len(P25_SYNC), 48)

    def test_sync_known_value(self):
        # 0x5575F5FF77FF as bits (MSB first)
        expected_hex = 0x5575F5FF77FF
        for i, bit in enumerate(P25_SYNC):
            expected = (expected_hex >> (47 - i)) & 1
            self.assertEqual(bit, expected,
                             msg=f"Bit {i} mismatch in P25 sync pattern")


class TestNIDParser(unittest.TestCase):
    def test_known_nac_duid(self):
        nac, duid = _parse_nid(_nid_bits(0x293, DUID_LDU1))
        self.assertEqual(nac, 0x293)
        self.assertEqual(duid, DUID_LDU1)

    def test_hdu_duid(self):
        _, duid = _parse_nid(_nid_bits(0x000, DUID_HDU))
        self.assertEqual(duid, DUID_HDU)

    def test_tdu_duid(self):
        _, duid = _parse_nid(_nid_bits(0x001, DUID_TDU))
        self.assertEqual(duid, DUID_TDU)


class TestHDUParser(unittest.TestCase):
    def test_hdu_fields_present(self):
        # Build a dummy 312-bit HDU payload
        # MI(72) + MFG(8) + ALGID(8=0x00 no encryption) + KID(16) + TGID(16) + ...
        payload = [0] * 312
        # ALGID=0x00 (no encryption) at bits 80-87
        fields = dict(_parse_hdu(payload))
        self.assertIn("MI", fields)
        self.assertIn("AlgID", fields)
        self.assertIn("TGID", fields)

    def test_hdu_too_short(self):
        fields = dict(_parse_hdu([0] * 50))
        self.assertIn("Error", fields)

    def test_hdu_algid_name(self):
        payload = [0] * 312
        # Set ALGID to 0x80 (AES) at bits 80-87
        algid = 0x80
        for i in range(8):
            payload[80 + i] = (algid >> (7 - i)) & 1
        fields = dict(_parse_hdu(payload))
        self.assertIn("AES", fields["AlgID"])


class TestLDU1LC(unittest.TestCase):
    def test_group_call_lc(self):
        lc_bits = [0] * 8  # LCF = 0x00 = Group Voice
        lc_bits += [0] * 8  # MFG
        dst = 5001
        src = 1234567
        lc_bits += int_to_bits(dst, 24)
        lc_bits += int_to_bits(src, 24)
        lc_bits += [0] * 8
        fields = dict(_parse_ldu1_lc(lc_bits))
        self.assertIn("Group Voice", fields["LCF"])
        self.assertEqual(fields["Dst ID"], str(dst))
        self.assertEqual(fields["Src ID"], str(src))

    def test_lc_too_short(self):
        fields = dict(_parse_ldu1_lc([0] * 10))
        self.assertIn("Error", fields)


class TestIMBEExtraction(unittest.TestCase):
    def test_extract_9_imbe_frames(self):
        # LDU payload = 9 × 160 bits
        payload = [0] * (9 * 160)
        frames = _extract_imbe_from_ldu(payload)
        self.assertEqual(len(frames), 9)

    def test_imbe_frame_length(self):
        payload = [0] * (9 * 160)
        frames = _extract_imbe_from_ldu(payload)
        for f in frames:
            self.assertEqual(len(f), 88)

    def test_short_payload_fewer_frames(self):
        payload = [0] * (3 * 160)
        frames = _extract_imbe_from_ldu(payload)
        self.assertEqual(len(frames), 3)


class TestP25Decoder(unittest.TestCase):
    def _make_decoder(self):
        return P25Decoder(max_sync_errors=4)

    def test_detects_tdu(self):
        dec = self._make_decoder()
        frame_bits = _make_p25_frame(DUID_TDU)
        frames = dec.push_bits(frame_bits)
        self.assertTrue(
            any(f.kind == FrameKind.CONTROL for f in frames),
            msg=f"Expected CONTROL frame for TDU, got: {frames}"
        )

    def test_detects_hdu(self):
        dec = self._make_decoder()
        payload = [0] * 312
        frame_bits = _make_p25_frame(DUID_HDU, payload)
        frames = dec.push_bits(frame_bits)
        self.assertTrue(
            any(f.kind == FrameKind.HEADER for f in frames),
            msg=f"Expected HEADER frame for HDU, got: {frames}"
        )

    def test_detects_ldu1(self):
        dec = self._make_decoder()
        ldu_payload = [0] * (9 * 160 + 72)
        frame_bits = _make_p25_frame(DUID_LDU1, ldu_payload)
        frames = dec.push_bits(frame_bits)
        voice = [f for f in frames if f.kind == FrameKind.VOICE]
        self.assertGreater(len(voice), 0,
                           msg="Expected at least one VOICE frame for LDU1")

    def test_ldu1_imbe_count(self):
        dec = self._make_decoder()
        ldu_payload = [0] * (9 * 160 + 72)
        frame_bits = _make_p25_frame(DUID_LDU1, ldu_payload)
        frames = dec.push_bits(frame_bits)
        for f in frames:
            if f.kind == FrameKind.VOICE:
                self.assertEqual(len(f.mbe_frames), 9,
                                 "LDU1 must carry 9 IMBE frames")

    def test_imbe_bit_lengths(self):
        dec = self._make_decoder()
        ldu_payload = [0] * (9 * 160 + 72)
        frame_bits = _make_p25_frame(DUID_LDU1, ldu_payload)
        frames = dec.push_bits(frame_bits)
        for f in frames:
            for mf in f.mbe_frames:
                self.assertEqual(len(mf.interleaved_bits), 88)
                self.assertEqual(len(mf.deinterleaved_bits), 88)

    def test_protocol_name(self):
        dec = self._make_decoder()
        frame_bits = _make_p25_frame(DUID_TDU)
        frames = dec.push_bits(frame_bits)
        for f in frames:
            self.assertEqual(f.protocol, "P25")

    def test_nac_in_header(self):
        dec = self._make_decoder()
        frame_bits = _make_p25_frame(DUID_TDU, nac=0xABC)
        frames = dec.push_bits(frame_bits)
        for f in frames:
            nac_fields = {k: v for k, v in f.header_fields if k == "NAC"}
            if nac_fields:
                self.assertIn("ABC", nac_fields["NAC"].upper())

    def test_sync_count_increments(self):
        dec = self._make_decoder()
        for _ in range(3):
            dec.push_bits(_make_p25_frame(DUID_TDU))
        self.assertGreater(dec.sync_count, 0)

    def test_reset(self):
        dec = self._make_decoder()
        dec.push_bits(_make_p25_frame(DUID_TDU))
        dec.reset()
        self.assertEqual(dec.sync_count, 0)


if __name__ == "__main__":
    unittest.main()
