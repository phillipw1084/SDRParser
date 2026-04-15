"""
DMR (Digital Mobile Radio) — ETSI TS 102 361-1
================================================

Frame structure (264 bits per 30 ms slot)
-----------------------------------------
::

    Bits   0-107 : Info Block 1  (108 bits)
    Bits 108-131 : SYNC pattern  ( 24 bits)  ← search target
    Bits 132-155 : EMB / reserved( 24 bits)
    Bits 156-263 : Info Block 2  (108 bits)

Sync patterns (24-bit hex)
--------------------------
* BS Voice: 0x755FD7    MS Voice: 0xDFF57D
* BS Data : 0xD7557F    MS Data : 0x7F7D5D
* BS Direct voice: 0x5D577F
* MS Direct voice: 0xF7FDD5

Voice Link Control (LC) header (72 bits in full LC)
----------------------------------------------------
::

    Bits  0-1  : FLCO  (2) — 0x0 = Group Voice Call
    Bit   2    : FID   (1)
    Bits  3-7  : SVCOPT(5) — service options
    Bits  8-27 : DST   (20) — destination address (group / unit)
    Bits 28-47 : SRC   (20) — source radio ID
    Bits 48-71 : RS (24) parity

AMBE+2 in a voice burst
-----------------------
Each 108-bit Info Block carries one 72-bit AMBE+2 frame embedded with a
simplified diagonal interleave (see :mod:`sdrparser.mbe.frames`).  There
are two voice frames per 30 ms burst.
"""

from __future__ import annotations

from typing import List, Optional

from sdrparser.dsp.demod import BitStreamBuffer, bits_to_int
from sdrparser.mbe.frames import MBEFrame, MBEType, INTERLEAVE_TABLES
from sdrparser.protocols.base import DecodedFrame, FrameKind, ProtocolDecoder

# ---------------------------------------------------------------------------
# DMR sync patterns (24 bits → bit lists, MSB first)
# ---------------------------------------------------------------------------

def _hex24_to_bits(value: int) -> List[int]:
    return [(value >> (23 - i)) & 1 for i in range(24)]


BS_VOICE_SYNC = _hex24_to_bits(0x755FD7)
MS_VOICE_SYNC = _hex24_to_bits(0xDFF57D)
BS_DATA_SYNC  = _hex24_to_bits(0xD7557F)
MS_DATA_SYNC  = _hex24_to_bits(0x7F7D5D)
BS_DIRECT_SYNC = _hex24_to_bits(0x5D577F)
MS_DIRECT_SYNC = _hex24_to_bits(0xF7FDD5)

VOICE_SYNCS = (BS_VOICE_SYNC, MS_VOICE_SYNC, BS_DIRECT_SYNC, MS_DIRECT_SYNC)
DATA_SYNCS  = (BS_DATA_SYNC, MS_DATA_SYNC)
ALL_SYNCS   = VOICE_SYNCS + DATA_SYNCS

SYNC_NAMES = {
    tuple(BS_VOICE_SYNC):  "BS Voice",
    tuple(MS_VOICE_SYNC):  "MS Voice",
    tuple(BS_DATA_SYNC):   "BS Data",
    tuple(MS_DATA_SYNC):   "MS Data",
    tuple(BS_DIRECT_SYNC): "BS Direct Voice",
    tuple(MS_DIRECT_SYNC): "MS Direct Voice",
}

# DMR burst layout
BURST_BITS   = 264
SYNC_OFFSET  = 108   # bit offset of the 24-bit SYNC word
SYNC_LEN     = 24
INFO1_LEN    = 108   # bits 0-107
INFO2_START  = 156   # bits 156-263
INFO2_LEN    = 108
EMB_OFFSET   = 132   # bits 132-155 (EMB / LCSS)
EMB_LEN      = 24

MAX_SYNC_ERRORS = 3  # tolerate up to 3 bit errors in sync pattern


# ---------------------------------------------------------------------------
# LC (Link Control) parsing
# ---------------------------------------------------------------------------

def _parse_lc(lc_bits: List[int]) -> List[tuple[str, str]]:
    """Parse 72-bit DMR Link Control word into labelled fields."""
    if len(lc_bits) < 72:
        return [("Error", "LC too short")]

    flco     = bits_to_int(lc_bits[0:2])
    fid      = lc_bits[2]
    svcopt   = bits_to_int(lc_bits[3:8])
    dst_id   = bits_to_int(lc_bits[8:28])
    src_id   = bits_to_int(lc_bits[28:48])

    flco_names = {0: "Group Voice Call", 1: "Group Data Call",
                  3: "Unit-to-Unit Voice", 4: "Unit-to-Unit Data"}
    flco_name = flco_names.get(flco, f"0x{flco:02X}")

    svcopt_str = (
        f"Priority={'High' if (svcopt >> 2) & 1 else 'Normal'}, "
        f"Emergency={'Yes' if (svcopt >> 7) & 1 else 'No'}"
    )

    return [
        ("FLCO",    flco_name),
        ("FID",     str(fid)),
        ("SvcOpt",  svcopt_str),
        ("Dst ID",  str(dst_id)),
        ("Src ID",  str(src_id)),
    ]


def _parse_csbk(bits: List[int]) -> List[tuple[str, str]]:
    """Parse a simplified CSBK (Control Signalling Block)."""
    if len(bits) < 96:
        return [("Error", "CSBK too short")]
    opcode = bits_to_int(bits[0:6])
    lb     = bits[6]
    pf     = bits[7]
    dst_id = bits_to_int(bits[24:48])
    src_id = bits_to_int(bits[48:72])

    opcode_names = {
        0x00: "BS Outbound Activation",
        0x01: "Unit-to-Unit Voice Channel User",
        0x03: "Channel Grant",
        0x28: "Aloha PDU",
    }
    return [
        ("CSBK Opcode", opcode_names.get(opcode, f"0x{opcode:02X}")),
        ("Last Block",  str(lb)),
        ("Protected",   str(pf)),
        ("Dst ID",      str(dst_id)),
        ("Src ID",      str(src_id)),
    ]


# ---------------------------------------------------------------------------
# AMBE+2 extraction from 108-bit info block
# ---------------------------------------------------------------------------

def _extract_ambe2_from_info_block(info_block: List[int]) -> List[int]:
    """Extract the 72-bit AMBE+2 payload from a 108-bit DMR info block.

    The 108-bit info block uses rate-3/4 BPTC for voice:  the first 72 bit
    positions (after deinterleaving) carry the AMBE+2 data; the remaining
    36 bits are parity.  For display purposes we return the raw 72 bits
    exactly as positioned in the info block (i.e., the interleaved form)
    and let :class:`~sdrparser.mbe.frames.MBEFrame` handle both views.
    """
    # Positions 0..71 of the info block are the interleaved AMBE+2 bits.
    # The diagonal interleave table covers exactly 72 positions out of 108.
    return list(info_block[:72])


# ---------------------------------------------------------------------------
# DMR decoder
# ---------------------------------------------------------------------------

class DMRDecoder(ProtocolDecoder):
    """Searches a bit stream for DMR burst sync patterns and decodes frames.

    Usage
    -----
    >>> from sdrparser.dsp.demod import BitStreamBuffer
    >>> dec = DMRDecoder()
    >>> frames = dec.push_bits(my_bit_list)
    """

    PROTOCOL_NAME = "DMR"

    def __init__(self, max_sync_errors: int = MAX_SYNC_ERRORS) -> None:
        super().__init__()
        self.max_sync_errors = max_sync_errors
        self._buf = BitStreamBuffer(max_bits=BURST_BITS * 8)

    # ------------------------------------------------------------------

    def push_bits(self, bits: List[int]) -> List[DecodedFrame]:
        self._buf.push_bits(bits)
        frames: List[DecodedFrame] = []

        while True:
            frame = self._try_decode()
            if frame is None:
                break
            frames.append(frame)

        return frames

    def _try_decode(self) -> Optional[DecodedFrame]:
        """Scan the buffer for a sync pattern and decode a burst."""
        best_pos   = -1
        best_sync  = None
        best_errs  = self.max_sync_errors + 1

        for sync in ALL_SYNCS:
            pos = self._find_sync_approx(sync)
            if pos < 0:
                continue
            errs = self._hamming_distance(
                sync, self._buf.peek(pos, SYNC_LEN)
            )
            if errs < best_errs:
                best_errs  = errs
                best_pos   = pos
                best_sync  = sync

        if best_pos < 0:
            return None

        # The SYNC is at offset SYNC_OFFSET inside the burst.
        # Actual burst start = best_pos - SYNC_OFFSET
        burst_start = best_pos - SYNC_OFFSET
        if burst_start < 0:
            # Not enough leading bits yet — discard up to sync position
            self._buf.consume(best_pos)
            return None

        if self._buf.bits_available() < burst_start + BURST_BITS:
            return None  # Wait for more bits

        self._buf.consume(burst_start)  # Align to burst start
        burst = self._buf.consume(BURST_BITS)
        return self._decode_burst(burst, best_sync)

    def _find_sync_approx(self, pattern: List[int]) -> int:
        """Find position of *pattern* allowing up to max_sync_errors errors."""
        plen = len(pattern)
        available = self._buf.bits_available()
        for i in range(available - plen + 1):
            window = self._buf.peek(i, plen)
            errs   = sum(a != b for a, b in zip(pattern, window))
            if errs <= self.max_sync_errors:
                return i
        return -1

    def _decode_burst(
        self, burst: List[int], sync_pattern: List[int]
    ) -> DecodedFrame:
        sync_name = SYNC_NAMES.get(tuple(sync_pattern), "Unknown")
        is_voice  = sync_pattern in VOICE_SYNCS

        info1 = burst[0:INFO1_LEN]
        emb   = burst[EMB_OFFSET:EMB_OFFSET + EMB_LEN]
        info2 = burst[INFO2_START:INFO2_START + INFO2_LEN]

        if is_voice:
            return self._decode_voice_burst(burst, info1, info2, sync_name)
        else:
            return self._decode_data_burst(burst, info1, emb, sync_name)

    def _decode_voice_burst(
        self,
        burst: List[int],
        info1: List[int],
        info2: List[int],
        sync_name: str,
    ) -> DecodedFrame:
        self.sync_count += 1
        self.sync_lost = False

        # Extract the two AMBE+2 payloads (interleaved form)
        ambe1_bits = _extract_ambe2_from_info_block(info1)
        ambe2_bits = _extract_ambe2_from_info_block(info2)

        mbe1 = MBEFrame.from_interleaved(
            "DMR", MBEType.AMBE2, self._frame_index,     ambe1_bits
        )
        mbe2 = MBEFrame.from_interleaved(
            "DMR", MBEType.AMBE2, self._frame_index + 1, ambe2_bits
        )
        self._frame_index += 2

        # Attempt to read embedded LC from the EMB region
        emb_data = burst[EMB_OFFSET:EMB_OFFSET + EMB_LEN]
        lc_bits  = burst[132:132 + 72] if len(burst) >= 204 else []
        header   = _parse_lc(lc_bits) if len(lc_bits) >= 72 else []
        header   = [("Sync", sync_name)] + header

        return DecodedFrame(
            protocol="DMR",
            kind=FrameKind.VOICE,
            header_fields=header,
            mbe_frames=[mbe1, mbe2],
            raw_header_bits=lc_bits,
            raw_bits=burst,
        )

    def _decode_data_burst(
        self,
        burst: List[int],
        info1: List[int],
        emb: List[int],
        sync_name: str,
    ) -> DecodedFrame:
        self.sync_count += 1
        self.sync_lost = False
        self._frame_index += 1

        # Treat the full 96-bit data payload as a CSBK if opcode is present
        data_bits = info1[:96] if len(info1) >= 96 else info1
        header    = _parse_csbk(data_bits)
        header    = [("Sync", sync_name)] + header

        return DecodedFrame(
            protocol="DMR",
            kind=FrameKind.DATA,
            header_fields=header,
            mbe_frames=[],
            raw_header_bits=data_bits,
            raw_bits=burst,
        )
