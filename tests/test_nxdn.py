"""
Tests for the NXDN decoder.
"""

import unittest
from sdrparser.protocols.nxdn import (
    NXDNDecoder,
    NXDN_FS_OUTBOUND,
    NXDN_FS_INBOUND,
    FRAME_BITS,
    LICH_OFFSET,
    RDCH_OFFSET,
    RDCH_LEN,
    _parse_lich,
    _parse_rdch,
    _extract_ambe2_from_rdch,
)
from sdrparser.protocols.base import FrameKind
from sdrparser.dsp.demod import int_to_bits


def _make_nxdn_frame(
    sync=None,
    rfct: int = 1,        # RTCH (trunked voice)
    ft: int = 1,          # Outbound
    option: int = 0,
    sf: int = 0,
    dst_id: int = 100,
    src_id: int = 200,
    msg_type: int = 0x01,  # Voice Channel User
    with_ambe: bool = True,
) -> list:
    """Build a synthetic 192-bit NXDN frame."""
    if sync is None:
        sync = NXDN_FS_OUTBOUND

    frame = [0] * FRAME_BITS

    # Frame sync (bits 0-15)
    for i, b in enumerate(sync):
        frame[i] = b

    # LICH byte (bits 16-23)
    lich_byte = (rfct << 6) | (ft << 4) | (option << 2) | (sf << 1)
    # Simple even parity
    parity = bin(lich_byte).count("1") % 2
    lich_byte |= parity
    for i in range(8):
        frame[LICH_OFFSET + i] = (lich_byte >> (7 - i)) & 1

    # LICH parity bits 24-31 (simplified: copy)
    for i in range(8):
        frame[24 + i] = frame[LICH_OFFSET + i]

    # RDCH (bits 32-191)
    rdch = [0] * RDCH_LEN
    # msg_type at bits 0-7
    for i in range(8):
        rdch[i] = (msg_type >> (7 - i)) & 1
    # version + flags = zeros
    # dst_id at bits 16-31
    dst_bits = int_to_bits(dst_id, 16)
    rdch[16:32] = dst_bits
    # src_id at bits 32-47
    src_bits = int_to_bits(src_id, 16)
    rdch[32:48] = src_bits

    # AMBE+2 at bits 80-151
    if with_ambe:
        ambe = [i % 2 for i in range(72)]
        rdch[80:152] = ambe

    for i, b in enumerate(rdch[:RDCH_LEN]):
        frame[RDCH_OFFSET + i] = b

    return frame


class TestSyncPatterns(unittest.TestCase):
    def test_outbound_length(self):
        self.assertEqual(len(NXDN_FS_OUTBOUND), 16)

    def test_inbound_length(self):
        self.assertEqual(len(NXDN_FS_INBOUND), 16)

    def test_syncs_different(self):
        self.assertNotEqual(NXDN_FS_OUTBOUND, NXDN_FS_INBOUND)


class TestLICHParser(unittest.TestCase):
    def test_parse_rtch(self):
        # RFCT=1 (RTCH), FT=1 (Outbound)
        lich_byte = (1 << 6) | (1 << 4)
        fields = dict(_parse_lich(lich_byte))
        self.assertIn("RTCH", fields["RFCT"])
        self.assertIn("Outbound", fields["FT"])

    def test_parse_conventional(self):
        lich_byte = (0 << 6) | (0 << 4)
        fields = dict(_parse_lich(lich_byte))
        self.assertIn("RCCH", fields["RFCT"])


class TestRDCHParser(unittest.TestCase):
    def test_voice_user_fields(self):
        rdch = [0] * RDCH_LEN
        msg_type = 0x01
        for i in range(8):
            rdch[i] = (msg_type >> (7 - i)) & 1
        dst = 1234
        src = 5678
        rdch[16:32] = int_to_bits(dst, 16)
        rdch[32:48] = int_to_bits(src, 16)
        fields = dict(_parse_rdch(rdch))
        self.assertEqual(fields["Dst ID"], str(dst))
        self.assertEqual(fields["Src ID"], str(src))
        self.assertIn("Voice", fields["Msg Type"])

    def test_rdch_too_short(self):
        fields = dict(_parse_rdch([0] * 10))
        self.assertIn("Error", fields)


class TestAMBE2Extraction(unittest.TestCase):
    def test_extract_returns_72_bits(self):
        rdch = [0] * RDCH_LEN
        ambe = [i % 2 for i in range(72)]
        rdch[80:152] = ambe
        result = _extract_ambe2_from_rdch(rdch)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 72)
        self.assertEqual(result, ambe)

    def test_extract_short_rdch_returns_none(self):
        result = _extract_ambe2_from_rdch([0] * 50)
        self.assertIsNone(result)


class TestNXDNDecoder(unittest.TestCase):
    def _make_decoder(self):
        return NXDNDecoder(max_sync_errors=2)

    def test_detects_outbound_voice(self):
        dec = self._make_decoder()
        frame = _make_nxdn_frame(sync=NXDN_FS_OUTBOUND, with_ambe=True)
        frames = dec.push_bits(frame)
        voice = [f for f in frames if f.kind == FrameKind.VOICE]
        self.assertGreater(len(voice), 0,
                           msg=f"Expected VOICE frame, got: {frames}")

    def test_detects_inbound_voice(self):
        dec = self._make_decoder()
        frame = _make_nxdn_frame(sync=NXDN_FS_INBOUND, with_ambe=True)
        frames = dec.push_bits(frame)
        self.assertGreater(len(frames), 0)

    def test_mbe_frame_present(self):
        dec = self._make_decoder()
        frame = _make_nxdn_frame(with_ambe=True)
        frames = dec.push_bits(frame)
        mbe_found = any(len(f.mbe_frames) > 0 for f in frames)
        self.assertTrue(mbe_found,
                        msg="Expected at least one MBE frame in voice burst")

    def test_mbe_length_72(self):
        dec = self._make_decoder()
        frame = _make_nxdn_frame(with_ambe=True)
        frames = dec.push_bits(frame)
        for f in frames:
            for mf in f.mbe_frames:
                self.assertEqual(len(mf.interleaved_bits), 72)
                self.assertEqual(len(mf.deinterleaved_bits), 72)

    def test_header_fields_present(self):
        dec = self._make_decoder()
        frame = _make_nxdn_frame(dst_id=999, src_id=111, with_ambe=False)
        frames = dec.push_bits(frame)
        for f in frames:
            field_dict = dict(f.header_fields)
            self.assertIn("Dst ID", field_dict)
            self.assertIn("Src ID", field_dict)

    def test_protocol_name(self):
        dec = self._make_decoder()
        frame = _make_nxdn_frame()
        frames = dec.push_bits(frame)
        for f in frames:
            self.assertEqual(f.protocol, "NXDN")

    def test_sync_count_increments(self):
        dec = self._make_decoder()
        for _ in range(3):
            dec.push_bits(_make_nxdn_frame())
        self.assertGreater(dec.sync_count, 0)

    def test_reset(self):
        dec = self._make_decoder()
        dec.push_bits(_make_nxdn_frame())
        dec.reset()
        self.assertEqual(dec.sync_count, 0)
        self.assertTrue(dec.sync_lost)

    def test_multiple_frames(self):
        dec = self._make_decoder()
        combined = []
        for _ in range(4):
            combined.extend(_make_nxdn_frame())
        frames = dec.push_bits(combined)
        self.assertGreaterEqual(len(frames), 4,
                                msg="Expected one frame per synthetic burst")


if __name__ == "__main__":
    unittest.main()
