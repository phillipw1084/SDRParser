"""
NXDN (Next Generation Digital Narrowband) decoder
==================================================

References: ETSI TS 102 166-1/2 (NXDN Common Air Interface)

Supported modes
---------------
* NXDN48  — 6.25 kHz channel spacing, 2 400 baud 4-FSK
* NXDN96  — 12.5 kHz channel spacing, 4 800 baud 4-FSK

Frame structure (192 bits per NXDN frame)
------------------------------------------
::

    Bits   0-15 : Frame Sync (FS)     — 16 bits
    Bits  16-31 : LICH                — 8 bits (+ 8 bits parity)
    Bits  32-191: Channel Data (RDCH) — 160 bits

LICH (Link Information CHannel, 8 bits)
----------------------------------------
::

    Bits 0-1 : RFCT (RF Channel Type)
               0 = Conventional (RCCH), 1 = Trunked (RTCH),
               2 = Trunked Supplementary (RTCH_C), 3 = Broadcast (BCCH)
    Bits 2-3 : FT  (Frame Type)
               0 = Inbound, 1 = Outbound, 2 = Inbound Superframe Start,
               3 = Outbound Superframe Start
    Bits 4-5 : Option
    Bit  6   : SF  (Superframe flag)
    Bit  7   : P   (Parity of bits 0-6)

RDCH (Radio Channel Data, 160 bits) — used for FACCH and VCCH
--------------------------------------------------------------
::

    Bits   0-7  : Message Type  (8)
    Bits   8-11 : Version       (4)
    Bits  12-15 : Flags         (4)
    Bits  16-31 : Dst ID        (16) — group or unit
    Bits  32-47 : Src ID        (16)
    Bits  48-79 : Additional / payload fields
    Bits  80-159: AMBE+2 voice  (2 × 40-bit half-frames = 80 bits)
                  OR remaining data depending on RFCT/FT

AMBE+2 in NXDN
--------------
NXDN carries 72-bit AMBE+2 frames split across two half-frames.  For
display purposes we show the complete 72-bit interleaved form and its
deinterleaved equivalent using the NXDN table from
:mod:`sdrparser.mbe.frames`.
"""

from __future__ import annotations

from typing import List, Optional

from sdrparser.dsp.demod import BitStreamBuffer, bits_to_int
from sdrparser.mbe.frames import MBEFrame, MBEType
from sdrparser.protocols.base import DecodedFrame, FrameKind, ProtocolDecoder

# ---------------------------------------------------------------------------
# NXDN frame sync patterns
# ---------------------------------------------------------------------------

def _sync_bits(value: int, width: int) -> List[int]:
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


# NXDN outbound and inbound frame sync patterns (16 bits)
NXDN_FS_OUTBOUND = _sync_bits(0x7654, 16)  # BS → MS
NXDN_FS_INBOUND  = _sync_bits(0x6543, 16)  # MS → BS
ALL_SYNCS        = [NXDN_FS_OUTBOUND, NXDN_FS_INBOUND]

FRAME_BITS       = 192
LICH_OFFSET      = 16   # bits
LICH_LEN         = 16   # 8 data + 8 parity bits
RDCH_OFFSET      = 32
RDCH_LEN         = 160
MAX_SYNC_ERRORS  = 2

# RFCT names
RFCT_NAMES = {0: "RCCH (Conventional)", 1: "RTCH (Trunked)",
              2: "RTCH_C (Supp)", 3: "BCCH (Broadcast)"}

# Frame type names
FT_NAMES = {0: "Inbound", 1: "Outbound",
            2: "Inbound Superframe Start", 3: "Outbound Superframe Start"}

# RDCH message type names
RDCH_MSG_NAMES = {
    0x01: "Voice Channel User",
    0x08: "Group Channel Grant",
    0x09: "Individual Channel Grant",
    0x18: "Registration Request",
    0x1C: "Call Alert",
    0x22: "Status Update",
}


# ---------------------------------------------------------------------------
# LICH parsing
# ---------------------------------------------------------------------------

def _parse_lich(lich_byte: int) -> List[tuple[str, str]]:
    """Parse the 8-bit LICH word."""
    rfct   = (lich_byte >> 6) & 0x3
    ft     = (lich_byte >> 4) & 0x3
    option = (lich_byte >> 2) & 0x3
    sf     = (lich_byte >> 1) & 0x1
    parity = lich_byte & 0x1
    return [
        ("RFCT",   RFCT_NAMES.get(rfct, str(rfct))),
        ("FT",     FT_NAMES.get(ft, str(ft))),
        ("Option", str(option)),
        ("SF",     str(sf)),
        ("Parity", str(parity)),
    ]


# ---------------------------------------------------------------------------
# RDCH parsing
# ---------------------------------------------------------------------------

def _parse_rdch(bits: List[int]) -> List[tuple[str, str]]:
    """Parse the 160-bit RDCH header fields."""
    if len(bits) < 48:
        return [("Error", "RDCH too short")]

    msg_type = bits_to_int(bits[0:8])
    version  = bits_to_int(bits[8:12])
    flags    = bits_to_int(bits[12:16])
    dst_id   = bits_to_int(bits[16:32])
    src_id   = bits_to_int(bits[32:48])

    msg_name = RDCH_MSG_NAMES.get(msg_type, f"0x{msg_type:02X}")

    return [
        ("Msg Type", msg_name),
        ("Version",  str(version)),
        ("Flags",    f"0x{flags:X}"),
        ("Dst ID",   str(dst_id)),
        ("Src ID",   str(src_id)),
    ]


# ---------------------------------------------------------------------------
# AMBE+2 extraction from RDCH payload
# ---------------------------------------------------------------------------

def _extract_ambe2_from_rdch(rdch_bits: List[int]) -> Optional[List[int]]:
    """Extract the 72-bit AMBE+2 frame from a voice RDCH payload.

    In voice frames the AMBE+2 data is located at bits 80-151 (72 bits).
    """
    if len(rdch_bits) < 152:
        return None
    return list(rdch_bits[80:152])


# ---------------------------------------------------------------------------
# NXDN decoder
# ---------------------------------------------------------------------------

class NXDNDecoder(ProtocolDecoder):
    """Searches a bit stream for NXDN frame sync and decodes frames.

    Usage
    -----
    >>> dec = NXDNDecoder()
    >>> frames = dec.push_bits(bits)
    """

    PROTOCOL_NAME = "NXDN"

    def __init__(self, max_sync_errors: int = MAX_SYNC_ERRORS) -> None:
        super().__init__()
        self.max_sync_errors = max_sync_errors
        self._buf = BitStreamBuffer(max_bits=FRAME_BITS * 8)

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
        best_pos  = -1
        best_sync = None
        best_errs = self.max_sync_errors + 1

        for sync in ALL_SYNCS:
            pos = self._find_sync_approx(sync)
            if pos < 0:
                continue
            errs = sum(
                a != b for a, b in zip(sync, self._buf.peek(pos, len(sync)))
            )
            if errs < best_errs:
                best_errs = errs
                best_pos  = pos
                best_sync = sync

        if best_pos < 0 or best_sync is None:
            return None

        if self._buf.bits_available() < best_pos + FRAME_BITS:
            return None

        self._buf.consume(best_pos)
        frame_bits = self._buf.consume(FRAME_BITS)
        return self._decode_frame(frame_bits, best_sync)

    def _find_sync_approx(self, pattern: List[int]) -> int:
        plen  = len(pattern)
        avail = self._buf.bits_available()
        for i in range(avail - plen + 1):
            window = self._buf.peek(i, plen)
            errs   = sum(a != b for a, b in zip(pattern, window))
            if errs <= self.max_sync_errors:
                return i
        return -1

    def _decode_frame(
        self, frame: List[int], sync_pattern: List[int]
    ) -> DecodedFrame:
        self.sync_count += 1
        self.sync_lost = False

        sync_name = ("Outbound" if sync_pattern == NXDN_FS_OUTBOUND
                     else "Inbound")

        # LICH (bits 16-23 are the 8 data bits; 24-31 are parity)
        lich_bits = frame[LICH_OFFSET:LICH_OFFSET + 8]
        lich_byte = bits_to_int(lich_bits)
        lich_fields = _parse_lich(lich_byte)

        rfct = (lich_byte >> 6) & 0x3
        ft   = (lich_byte >> 4) & 0x3

        rdch_bits = frame[RDCH_OFFSET:RDCH_OFFSET + RDCH_LEN]
        rdch_fields = _parse_rdch(rdch_bits)

        # Is this a voice frame?  RFCT ∈ {0,1,2} and FT indicates voice
        is_voice = (rfct in (0, 1, 2)) and (len(rdch_bits) >= 152)
        ambe_bits = _extract_ambe2_from_rdch(rdch_bits) if is_voice else None

        mbe_frames: List[MBEFrame] = []
        if ambe_bits and len(ambe_bits) == 72:
            mbe = MBEFrame.from_interleaved(
                "NXDN", MBEType.AMBE2, self._frame_index, ambe_bits
            )
            mbe_frames.append(mbe)
            self._frame_index += 1

        kind = FrameKind.VOICE if mbe_frames else FrameKind.CONTROL
        header = (
            [("Sync", sync_name)]
            + lich_fields
            + rdch_fields
        )

        return DecodedFrame(
            protocol="NXDN",
            kind=kind,
            header_fields=header,
            mbe_frames=mbe_frames,
            raw_header_bits=lich_bits + rdch_bits[:48],
            raw_bits=frame,
        )
