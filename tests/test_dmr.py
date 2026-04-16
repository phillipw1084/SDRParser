"""
Tests for the DMR decoder.
"""

import unittest
from sdrparser.protocols.dmr import (
    DMRDecoder,
    BS_VOICE_SYNC,
    MS_VOICE_SYNC,
    BS_DATA_SYNC,
    BURST_BITS,
    SYNC_OFFSET,
    _parse_lc,
    _parse_csbk,
    _bptc_deinterleave_196,
    _bptc_extract_96,
    _bptc_extract_96_corrected,
    _hamming_13_9_correct,
    _crc16_ccitt_dmr,
    _hamming_15_11_correct,
)
from sdrparser.protocols.base import FrameKind
from sdrparser.dsp.demod import bits_to_int


def _make_voice_burst(sync=None, lc_dst=1234, lc_src=5678) -> list:
    """Build a synthetic 264-bit DMR voice burst."""
    if sync is None:
        sync = BS_VOICE_SYNC

    burst = [0] * BURST_BITS

    # Place sync at bits 108-131
    for i, b in enumerate(sync):
        burst[SYNC_OFFSET + i] = b

    # Embed a simple LC in bits 132-203 (72 bits)
    # FLCO=0, FID=0, SVCOPT=0, DST=lc_dst (20 bits), SRC=lc_src (20 bits)
    dst_bits = [(lc_dst >> (19 - i)) & 1 for i in range(20)]
    src_bits = [(lc_src >> (19 - i)) & 1 for i in range(20)]
    lc_payload = [0] * 8 + dst_bits + src_bits + [0] * 44
    for i, b in enumerate(lc_payload[:72]):
        burst[132 + i] = b

    # Info block 1: set some AMBE+2 bits
    for i in range(72):
        burst[i] = i % 2

    return burst


class TestSyncPatterns(unittest.TestCase):
    def test_bs_voice_sync_length(self):
        self.assertEqual(len(BS_VOICE_SYNC), 24)

    def test_ms_voice_sync_length(self):
        self.assertEqual(len(MS_VOICE_SYNC), 24)

    def test_bs_data_sync_length(self):
        self.assertEqual(len(BS_DATA_SYNC), 24)

    def test_syncs_are_different(self):
        self.assertNotEqual(BS_VOICE_SYNC, MS_VOICE_SYNC)
        self.assertNotEqual(BS_VOICE_SYNC, BS_DATA_SYNC)


class TestLCParser(unittest.TestCase):
    def test_group_voice_call(self):
        # Build a minimal 72-bit LC: FLCO=0 (group), DST=100, SRC=200
        dst = 100
        src = 200
        lc_bits = [0, 0]  # FLCO
        lc_bits += [0]     # FID
        lc_bits += [0] * 5  # SVCOPT
        lc_bits += [(dst >> (19 - i)) & 1 for i in range(20)]
        lc_bits += [(src >> (19 - i)) & 1 for i in range(20)]
        lc_bits += [0] * 24  # RS parity padding

        fields = dict(_parse_lc(lc_bits))
        self.assertIn("Group Voice Call", fields["FLCO"])
        self.assertEqual(fields["Dst ID"], "100")
        self.assertEqual(fields["Src ID"], "200")

    def test_lc_too_short(self):
        fields = dict(_parse_lc([0] * 10))
        self.assertIn("Error", fields)


class TestCSBKParser(unittest.TestCase):
    def test_short_csbk(self):
        fields = dict(_parse_csbk([0] * 10))
        self.assertIn("Error", fields)

    def test_minimal_csbk(self):
        # 96 bits of zeros → opcode=0 = BS Outbound Activation
        fields = dict(_parse_csbk([0] * 96))
        self.assertIn("Outbound", fields["CSBK Opcode"])


class TestDMRDecoder(unittest.TestCase):
    def _make_decoder(self):
        return DMRDecoder(max_sync_errors=2)

    def test_detects_voice_burst(self):
        """Decoder must detect a synthetic voice burst and return a VOICE frame."""
        dec = self._make_decoder()
        burst = _make_voice_burst(sync=BS_VOICE_SYNC)
        frames = dec.push_bits(burst)
        voice_frames = [f for f in frames if f.kind == FrameKind.VOICE]
        self.assertTrue(len(voice_frames) >= 1,
                        msg=f"Expected a VOICE frame, got: {frames}")

    def test_returns_three_mbe_frames_per_voice_burst(self):
        dec = self._make_decoder()
        burst = _make_voice_burst(sync=BS_VOICE_SYNC)
        frames = dec.push_bits(burst)
        voice = [f for f in frames if f.kind == FrameKind.VOICE]
        self.assertGreater(len(voice), 0)
        self.assertEqual(len(voice[0].mbe_frames), 3,
                         msg="Each DMR voice payload burst should carry 3 AMBE+2 frames")

    def test_mbe_frames_have_correct_length(self):
        dec = self._make_decoder()
        burst = _make_voice_burst()
        frames = dec.push_bits(burst)
        for f in frames:
            for mf in f.mbe_frames:
                self.assertEqual(len(mf.interleaved_bits), 72)
                self.assertEqual(len(mf.deinterleaved_bits), 72)

    def test_sync_count_increments(self):
        dec = self._make_decoder()
        burst = _make_voice_burst()
        dec.push_bits(burst)
        self.assertGreater(dec.sync_count, 0)

    def test_protocol_name(self):
        dec = self._make_decoder()
        burst = _make_voice_burst()
        frames = dec.push_bits(burst)
        for f in frames:
            self.assertEqual(f.protocol, "DMR")

    def test_ms_voice_sync_detected(self):
        dec = self._make_decoder()
        burst = _make_voice_burst(sync=MS_VOICE_SYNC)
        frames = dec.push_bits(burst)
        self.assertTrue(any(f.kind == FrameKind.VOICE for f in frames))

    def test_reset_clears_state(self):
        dec = self._make_decoder()
        burst = _make_voice_burst()
        dec.push_bits(burst)
        self.assertGreater(dec.sync_count, 0)
        dec.reset()
        self.assertEqual(dec.sync_count, 0)
        self.assertTrue(dec.sync_lost)

    def test_multiple_consecutive_bursts(self):
        dec = self._make_decoder()
        combined = []
        for _ in range(3):
            combined.extend(_make_voice_burst())
        frames = dec.push_bits(combined)
        voice = [f for f in frames if f.kind == FrameKind.VOICE]
        self.assertGreaterEqual(len(voice), 3,
                                msg="Expected one voice frame per burst")


class TestDMRBPTCHelpers(unittest.TestCase):
    def test_bptc_deinterleave_length(self):
        data = [0] * 196
        out = _bptc_deinterleave_196(data)
        self.assertEqual(len(out), 196)

    def test_bptc_extract_96_length(self):
        out = _bptc_extract_96([0] * 196)
        self.assertEqual(len(out), 96)

    def test_bptc_extract_96_corrected_length(self):
        out, ok = _bptc_extract_96_corrected([0] * 196)
        self.assertTrue(ok)
        self.assertEqual(len(out), 96)


class TestDMRFECAndCRC(unittest.TestCase):
    def test_hamming_13_9_corrects_single_bit_error(self):
        # All-zero codeword is valid; single-bit flips should be corrected.
        cw = [0] * 13
        cw[4] = 1
        data, ok = _hamming_13_9_correct(cw)
        self.assertTrue(ok)
        self.assertEqual(data, [0] * 9)

    def test_hamming_15_11_corrects_single_bit_error(self):
        # All-zero 15-bit codeword is valid; single-bit flips should be corrected.
        cw = [0] * 15
        cw[7] = 1
        data, ok = _hamming_15_11_correct(cw)
        self.assertTrue(ok)
        self.assertEqual(data, [0] * 11)

    def test_crc16_changes_when_payload_changes(self):
        a = [0] * 80
        b = [0] * 80
        b[10] = 1
        self.assertNotEqual(_crc16_ccitt_dmr(a), _crc16_ccitt_dmr(b))

if __name__ == "__main__":
    unittest.main()
