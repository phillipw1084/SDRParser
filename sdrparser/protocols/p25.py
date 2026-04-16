"""
P25 (Project 25 / APCO-25) Phase 1 decoder
===========================================

Reference: TIA-102.BAAA (FDMA Common Air Interface)

Frame sync
----------
48-bit pattern: 0x5575F5FF77FF (bits 0-47 of every P25 frame header).

NID (Network Identifier, 64 bits after frame sync)
---------------------------------------------------
::

    Bits 0-11  : NAC  (12) — Network Access Code
    Bits 12-15 : DUID (4)  — Data Unit ID

DUID values
-----------
=====  ====
0x0    HDU  — Header Data Unit
0x3    TDU  — Terminator (no LC)
0x5    LDU1 — Logical Data Unit 1  (voice + LC)
0xA    LDU2 — Logical Data Unit 2  (voice + encryption)
0xB    TDULC — Terminator with LC
0xC    PDU  — Packet Data Unit
0xF    TSDU — Trunking Signalling Data Unit
=====  ====

HDU header (after NID)
-----------------------
::

    MI    (72)  — Message Indicator
    MFG   (8)   — Manufacturer ID
    ALGID (8)   — Algorithm ID
    KID   (16)  — Key ID
    TGID  (16)  — Talkgroup ID

LDU1 Link Control word (9 groups × 9 bytes = carried interleaved across
the LDU superframe, simplified extraction below)
::

    LCF   (8)   — Link Control Format
    MFG   (8)   — Manufacturer ID
    SRC   (24)  — Source radio ID
    DST   (24)  — Destination ID (group or unit)

IMBE frames
-----------
Each LDU carries 9 × 88-bit IMBE voice frames.  The bits are extracted
from the LDU's 9 voice codewords and deinterleaved with the P25 IMBE
table (see :mod:`sdrparser.mbe.frames`).
"""

from __future__ import annotations

from typing import List, Optional

from sdrparser.dsp.demod import BitStreamBuffer, bits_to_int
from sdrparser.mbe.frames import MBEFrame, MBEType
from sdrparser.protocols.base import DecodedFrame, FrameKind, ProtocolDecoder

# ---------------------------------------------------------------------------
# P25 frame sync pattern (48 bits)
# ---------------------------------------------------------------------------

def _sync_bits(value: int, width: int) -> List[int]:
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


P25_SYNC          = _sync_bits(0x5575F5FF77FF, 48)
P25_SYNC_LEN      = 48
P25_NID_LEN       = 64    # NAC (12) + DUID (4) + RS parity (48)
P25_HDR_LEN       = P25_SYNC_LEN + P25_NID_LEN

MAX_SYNC_ERRORS   = 4     # tolerate up to 4 bit errors in 48-bit sync

# DUID identifiers
DUID_HDU   = 0x0
DUID_TDU   = 0x3
DUID_LDU1  = 0x5
DUID_LDU2  = 0xA
DUID_TDULC = 0xB
DUID_PDU   = 0xC
DUID_TSDU  = 0xF

DUID_NAMES = {
    DUID_HDU:   "HDU  (Header Data Unit)",
    DUID_TDU:   "TDU  (Terminator)",
    DUID_LDU1:  "LDU1 (Voice + LC)",
    DUID_LDU2:  "LDU2 (Voice + Encrypt)",
    DUID_TDULC: "TDULC (Terminator + LC)",
    DUID_PDU:   "PDU  (Packet Data)",
    DUID_TSDU:  "TSDU (Trunking Signal)",
}

# LDU frame sizes (bits after frame sync + NID)
LDU_PAYLOAD_BITS = 1440   # 9 voice codewords × 160 bits each
HDU_PAYLOAD_BITS = 312    # MI + MFG + ALGID + KID + TGID + RS

ALGID_NAMES = {
    0x00: "No encryption",
    0x01: "DES-OFB",
    0x02: "2-key Triple DES",
    0x03: "3-key Triple DES",
    0x05: "AES-256",
    0x80: "AES",
    0x81: "BATON",
    0x84: "FIREFLY",
}


# ---------------------------------------------------------------------------
# Header parsers
# ---------------------------------------------------------------------------

def _parse_nid(bits: List[int]) -> tuple[int, int]:
    """Return (NAC, DUID) from 64-bit NID (ignoring RS parity)."""
    nac  = bits_to_int(bits[0:12])
    duid = bits_to_int(bits[12:16])
    return nac, duid


def _parse_hdu(bits: List[int]) -> List[tuple[str, str]]:
    """Parse HDU payload (312 bits after NID)."""
    if len(bits) < 312:
        return [("Error", f"HDU payload too short ({len(bits)} bits)")]
    mi    = bits_to_int(bits[0:72])
    mfg   = bits_to_int(bits[72:80])
    algid = bits_to_int(bits[80:88])
    kid   = bits_to_int(bits[88:104])
    tgid  = bits_to_int(bits[104:120])
    return [
        ("MI",     f"0x{mi:018X}"),
        ("MfgID",  f"0x{mfg:02X}"),
        ("AlgID",  ALGID_NAMES.get(algid, f"0x{algid:02X}")),
        ("KeyID",  f"0x{kid:04X}"),
        ("TGID",   str(tgid)),
    ]


def _parse_ldu1_lc(lc_bits: List[int]) -> List[tuple[str, str]]:
    """Parse LDU1 Link Control word (72 bits)."""
    if len(lc_bits) < 72:
        return [("Error", "LC too short")]
    lcf   = bits_to_int(lc_bits[0:8])
    mfg   = bits_to_int(lc_bits[8:16])
    dst   = bits_to_int(lc_bits[16:40])
    src   = bits_to_int(lc_bits[40:64])

    lcf_names = {
        0x00: "Group Voice Channel User",
        0x03: "Unit-to-Unit Voice Channel User",
        0x10: "Telephone Interconnect Voice Channel User",
    }
    return [
        ("LCF",    lcf_names.get(lcf, f"0x{lcf:02X}")),
        ("MfgID",  f"0x{mfg:02X}"),
        ("Dst ID", str(dst)),
        ("Src ID", str(src)),
    ]


def _parse_ldu2_enc(bits: List[int]) -> List[tuple[str, str]]:
    """Parse LDU2 encryption fields (after 9 voice codewords)."""
    if len(bits) < 88:
        return [("Encrypt", "Fields unavailable")]
    algid = bits_to_int(bits[0:8])
    kid   = bits_to_int(bits[8:24])
    mi    = bits_to_int(bits[24:96]) if len(bits) >= 96 else 0
    return [
        ("AlgID",  ALGID_NAMES.get(algid, f"0x{algid:02X}")),
        ("KeyID",  f"0x{kid:04X}"),
        ("MI",     f"0x{mi:018X}"),
    ]


# ---------------------------------------------------------------------------
# IMBE extraction from LDU codewords
# ---------------------------------------------------------------------------

# An LDU carries 9 voice codewords.  Each codeword is 144 bits, from which
# an 88-bit IMBE frame is extracted at specific positions.  The remaining
# bits carry error-correction data and overhead.
# Voice codeword positions within 144 bits (TIA-102.BAAA Table 7.5):
_IMBE_POSITIONS_IN_CODEWORD = list(range(88))   # first 88 bits are IMBE


def _extract_imbe_from_ldu(payload: List[int]) -> List[List[int]]:
    """Extract up to 9 raw 88-bit IMBE frames from an LDU payload.

    The LDU payload (1440 bits) contains 9 voice codewords of 160 bits
    each.  Within each codeword the first 88 bits carry the IMBE frame
    (interleaved form).
    """
    frames: List[List[int]] = []
    codeword_size = 160
    for i in range(9):
        start = i * codeword_size
        end   = start + codeword_size
        if end > len(payload):
            break
        codeword = payload[start:end]
        # First 88 bits are the interleaved IMBE frame
        imbe_bits = codeword[:88]
        if len(imbe_bits) == 88:
            frames.append(imbe_bits)
    return frames


# ---------------------------------------------------------------------------
# P25 decoder
# ---------------------------------------------------------------------------

class P25Decoder(ProtocolDecoder):
    """Searches a bit stream for P25 frame sync and decodes frames.

    Usage
    -----
    >>> dec = P25Decoder()
    >>> frames = dec.push_bits(bits_from_demodulator)
    """

    PROTOCOL_NAME = "P25"

    def __init__(self, max_sync_errors: int = MAX_SYNC_ERRORS) -> None:
        super().__init__()
        self.max_sync_errors = max_sync_errors
        self._buf = BitStreamBuffer(max_bits=8192)

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
        pos = self._find_sync()
        if pos < 0:
            return None

        # Check we have enough bits for at least the NID
        if self._buf.bits_available() < pos + P25_HDR_LEN:
            return None

        # Consume bits up to and including the NID
        self._buf.consume(pos)                      # skip pre-sync garbage
        sync_bits = self._buf.consume(P25_SYNC_LEN) # consume sync
        nid_bits  = self._buf.consume(P25_NID_LEN)  # consume NID

        nac, duid = _parse_nid(nid_bits)
        return self._decode_payload(nac, duid, sync_bits + nid_bits)

    def _find_sync(self) -> int:
        return self._buf.find_pattern_approx(P25_SYNC, self.max_sync_errors)

    def _decode_payload(
        self,
        nac: int,
        duid: int,
        header_bits: List[int],
    ) -> DecodedFrame:
        self.sync_count += 1
        self.sync_lost = False

        duid_name = DUID_NAMES.get(duid, f"0x{duid:X}")

        base_fields = [
            ("DUID", duid_name),
            ("NAC",  f"0x{nac:03X}"),
        ]

        if duid == DUID_HDU:
            return self._decode_hdu(base_fields, header_bits)

        if duid in (DUID_LDU1, DUID_LDU2):
            return self._decode_ldu(duid, base_fields, header_bits)

        if duid in (DUID_TDU, DUID_TDULC):
            return DecodedFrame(
                protocol="P25",
                kind=FrameKind.CONTROL,
                header_fields=base_fields + [("Info", "Call terminator")],
                mbe_frames=[],
                raw_header_bits=header_bits,
                raw_bits=header_bits,
            )

        return DecodedFrame(
            protocol="P25",
            kind=FrameKind.DATA,
            header_fields=base_fields,
            mbe_frames=[],
            raw_header_bits=header_bits,
            raw_bits=header_bits,
        )

    def _decode_hdu(
        self,
        base_fields: List[tuple[str, str]],
        header_bits: List[int],
    ) -> DecodedFrame:
        # Wait for HDU payload
        if self._buf.bits_available() < HDU_PAYLOAD_BITS:
            return DecodedFrame(
                protocol="P25",
                kind=FrameKind.HEADER,
                header_fields=base_fields + [("Info", "HDU (partial)")],
                mbe_frames=[],
                raw_header_bits=header_bits,
                raw_bits=header_bits,
            )
        payload = self._buf.consume(HDU_PAYLOAD_BITS)
        hdu_fields = _parse_hdu(payload)
        self._frame_index += 1
        return DecodedFrame(
            protocol="P25",
            kind=FrameKind.HEADER,
            header_fields=base_fields + hdu_fields,
            mbe_frames=[],
            raw_header_bits=header_bits + payload,
            raw_bits=header_bits + payload,
        )

    def _decode_ldu(
        self,
        duid: int,
        base_fields: List[tuple[str, str]],
        header_bits: List[int],
    ) -> DecodedFrame:
        # Wait for full LDU payload
        if self._buf.bits_available() < LDU_PAYLOAD_BITS:
            return DecodedFrame(
                protocol="P25",
                kind=FrameKind.VOICE,
                header_fields=base_fields + [("Info", "LDU (partial)")],
                mbe_frames=[],
                raw_header_bits=header_bits,
                raw_bits=header_bits,
            )

        payload = self._buf.consume(LDU_PAYLOAD_BITS)

        # Extract IMBE frames
        raw_imbe_list = _extract_imbe_from_ldu(payload)
        mbe_frames = [
            MBEFrame.from_interleaved(
                "P25", MBEType.IMBE, self._frame_index + i, imbe_bits
            )
            for i, imbe_bits in enumerate(raw_imbe_list)
        ]
        self._frame_index += len(mbe_frames)

        # Parse LC or encryption header from trailing bits
        trailer = payload[9 * 160:]          # bits after 9 codewords
        if duid == DUID_LDU1:
            lc_fields = _parse_ldu1_lc(trailer[:72])
            header_fields = base_fields + lc_fields
            raw_header_bits = header_bits + trailer[:72]
            kind = FrameKind.VOICE
        else:
            enc_fields = _parse_ldu2_enc(trailer[:96])
            header_fields = base_fields + enc_fields
            raw_header_bits = header_bits + trailer[:96]
            kind = FrameKind.VOICE

        return DecodedFrame(
            protocol="P25",
            kind=kind,
            header_fields=header_fields,
            mbe_frames=mbe_frames,
            raw_header_bits=raw_header_bits,
            raw_bits=header_bits + payload,
        )
