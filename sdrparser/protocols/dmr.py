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
from sdrparser.mbe.frames import MBEFrame, MBEType
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

MAX_SYNC_ERRORS = 1  # tighter lock to reduce false-positive burst decodes


# Hamming (13,9,3) parity-check matrix (from DSD/ETSI implementation)
_HAMMING_13_9_H = (
    (1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0),
    (1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0),
    (1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0),
    (1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1),
)

# Hamming (15,11,3) generator rows (systematic) from DSD/ETSI implementation.
_HAMMING_15_11_G = (
    "100000000001111",
    "010000000001110",
    "001000000001101",
    "000100000001100",
    "000010000001011",
    "000001000001010",
    "000000100001001",
    "000000010000111",
    "000000001000110",
    "000000000100101",
    "000000000010011",
)

_HAMMING_15_11_CODEBOOK: dict[tuple[int, ...], List[int]] = {}


def _init_hamming_15_11_codebook() -> None:
    if _HAMMING_15_11_CODEBOOK:
        return
    grows = [[int(ch) for ch in row] for row in _HAMMING_15_11_G]
    for msg in range(1 << 11):
        cw = [0] * 15
        data = [((msg >> (10 - i)) & 1) for i in range(11)]
        for i, bit in enumerate(data):
            if bit:
                row = grows[i]
                for j in range(15):
                    cw[j] ^= row[j]
        _HAMMING_15_11_CODEBOOK[tuple(cw)] = data


def _hamming_15_11_correct(codeword: List[int]) -> tuple[List[int], bool]:
    """Correct one 15-bit Hamming(15,11,3) codeword; return (data11, ok)."""
    if len(codeword) != 15:
        return [], False

    _init_hamming_15_11_codebook()
    cw = [b & 1 for b in codeword]
    hit = _HAMMING_15_11_CODEBOOK.get(tuple(cw))
    if hit is not None:
        return list(hit), True

    candidate: Optional[List[int]] = None
    for i in range(15):
        test = list(cw)
        test[i] ^= 1
        hit = _HAMMING_15_11_CODEBOOK.get(tuple(test))
        if hit is None:
            continue
        if candidate is not None and candidate != hit:
            return [], False
        candidate = list(hit)

    if candidate is None:
        return [], False
    return candidate, True


def _crc16_ccitt_dmr(bits: List[int]) -> int:
    """Compute DMR CRC-CCITT over a bit sequence (MSB-first)."""
    crc = 0
    poly = 0x1021
    for bit in bits:
        in_bit = bit & 1
        if (((crc >> 15) & 1) ^ in_bit) != 0:
            crc = ((crc << 1) ^ poly) & 0xFFFF
        else:
            crc = (crc << 1) & 0xFFFF
    return crc ^ 0xFFFF


def _hamming_13_9_correct(codeword: List[int]) -> tuple[List[int], bool]:
    """Correct one 13-bit Hamming(13,9,3) codeword; return (data9, ok)."""
    if len(codeword) != 13:
        return [], False

    cw = [b & 1 for b in codeword]
    syndrome = [sum(cw[i] * _HAMMING_13_9_H[r][i] for i in range(13)) & 1 for r in range(4)]
    if any(syndrome):
        error_pos = -1
        for i in range(13):
            col = [_HAMMING_13_9_H[r][i] for r in range(4)]
            if col == syndrome:
                error_pos = i
                break
        if error_pos < 0:
            return [], False
        cw[error_pos] ^= 1

    return cw[:9], True


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
        f"Emergency={'Yes' if (svcopt >> 4) & 1 else 'No'}, "
        f"Privacy={'On' if (svcopt & 1) else 'Off'}"
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


def _extract_ambe2_triplet_from_burst(burst: List[int]) -> List[List[int]]:
    """Extract 3 AMBE+2 frames from one 264-bit DMR payload burst.

    Alignment follows dsd-fme payload handling (without CACH):
    - AMBE1: dibits 0..35
    - AMBE2: dibits 36..53 and 78..95 (split around 24 dibits sync/EMB)
    - AMBE3: dibits 96..131
    """

    if len(burst) < BURST_BITS:
        return []

    dibits: List[int] = []
    for i in range(0, BURST_BITS, 2):
        dibits.append(((burst[i] & 1) << 1) | (burst[i + 1] & 1))

    def dibits_to_bits(db: List[int]) -> List[int]:
        out: List[int] = []
        for d in db:
            out.append((d >> 1) & 1)
            out.append(d & 1)
        return out

    ambe1 = dibits_to_bits(dibits[0:36])
    ambe2 = dibits_to_bits(dibits[36:54] + dibits[78:96])
    ambe3 = dibits_to_bits(dibits[96:132])
    return [ambe1, ambe2, ambe3]


def _bptc_deinterleave_196(input_bits: List[int]) -> List[int]:
    """DMR BPTC (196,96) deinterleave using ETSI/DSD index mapping."""
    if len(input_bits) != 196:
        return []
    out = [0] * 196
    for i, bit in enumerate(input_bits):
        out[(i * 13) % 196] = bit & 1
    return out


def _bptc_extract_96(deinterleaved_bits: List[int]) -> List[int]:
    """Extract 96 payload bits from deinterleaved BPTC matrix.

    This mirrors matrix placement/extraction used by DSD-family decoders,
    but does not yet apply row/column Hamming correction.
    """
    if len(deinterleaved_bits) != 196:
        return []

    # Build 13x15 matrix, skipping leading R(3) bit.
    matrix = [[0] * 15 for _ in range(13)]
    k = 1
    for r in range(13):
        for c in range(15):
            matrix[r][c] = deinterleaved_bits[k] & 1
            k += 1

    extracted: List[int] = []
    # Row 0, cols 3..10 (8 bits)
    for c in range(3, 11):
        extracted.append(matrix[0][c])
    # Rows 1..8, cols 0..10 (88 bits)
    for r in range(1, 9):
        for c in range(11):
            extracted.append(matrix[r][c])

    return extracted if len(extracted) == 96 else []


def _bptc_extract_96_corrected(deinterleaved_bits: List[int]) -> tuple[List[int], bool]:
    """Extract 96 bits after dsd-fme style BPTC row/column correction."""
    if len(deinterleaved_bits) != 196:
        return [], False

    # Build 13x15 matrix, skipping leading R(3) bit.
    matrix = [[0] * 15 for _ in range(13)]
    k = 1
    for r in range(13):
        for c in range(15):
            matrix[r][c] = deinterleaved_bits[k] & 1
            k += 1

    # First pass: row Hamming(15,11) for rows 0..8.
    for r in range(9):
        data11, ok = _hamming_15_11_correct(matrix[r])
        if not ok:
            return [], False
        for c in range(11):
            matrix[r][c] = data11[c]

    # Second pass: column Hamming(13,9), then repeat once (as in dsd-fme).
    for _ in range(2):
        for c in range(15):
            codeword = [matrix[r][c] for r in range(13)]
            data9, ok = _hamming_13_9_correct(codeword)
            if not ok:
                return [], False
            for r in range(9):
                matrix[r][c] = data9[r]

    extracted: List[int] = []
    for c in range(3, 11):
        extracted.append(matrix[0][c])
    for r in range(1, 9):
        for c in range(11):
            extracted.append(matrix[r][c])

    return (extracted, True) if len(extracted) == 96 else ([], False)


def _extract_data_196_from_burst(burst: List[int]) -> List[int]:
    """Extract the 196 BPTC data bits from a DMR data burst payload.

    For 132 dibits payload layout: 49 data + 5 slot type + 24 sync +
    5 slot type + 49 data.
    """
    if len(burst) < BURST_BITS:
        return []

    part_a = burst[0:98]      # 49 dibits
    part_b = burst[166:264]   # 49 dibits
    bits = part_a + part_b
    return bits if len(bits) == 196 else []


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
        self._tx_counter = 0
        self._active_tx_meta: dict[str, str] = {}

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
        best_pos, best_pat_idx, _best_errs = self._buf.find_best_pattern_approx(
            list(ALL_SYNCS), self.max_sync_errors
        )
        best_sync = ALL_SYNCS[best_pat_idx] if best_pat_idx >= 0 else None

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

        # dsd-fme style alignment yields 3 AMBE frames per payload burst.
        ambe_triplet = _extract_ambe2_triplet_from_burst(burst)
        mbe_frames: List[MBEFrame] = []
        for i, bits in enumerate(ambe_triplet):
            if len(bits) == 72:
                mbe_frames.append(
                    MBEFrame.from_interleaved(
                        "DMR", MBEType.AMBE2, self._frame_index + i, bits
                    )
                )
        self._frame_index += len(mbe_frames)

        emb_data = burst[EMB_OFFSET:EMB_OFFSET + EMB_LEN]
        header = [
            ("Sync", sync_name),
            ("VoiceFrames", str(len(mbe_frames))),
            ("PI", self._active_tx_meta.get("PI", "Unknown")),
        ]
        if self._active_tx_meta.get("Tx"):
            header.append(("Tx", self._active_tx_meta["Tx"]))
        raw_header_bits = (
            burst[SYNC_OFFSET:SYNC_OFFSET + SYNC_LEN]
            + emb_data
        )

        return DecodedFrame(
            protocol="DMR",
            kind=FrameKind.VOICE,
            header_fields=header,
            mbe_frames=mbe_frames,
            raw_header_bits=raw_header_bits,
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

        data_196 = _extract_data_196_from_burst(burst)
        deint_196 = _bptc_deinterleave_196(data_196) if data_196 else []
        data_bits = _bptc_extract_96(deint_196) if deint_196 else []
        corrected_bits, corrected_ok = (
            _bptc_extract_96_corrected(deint_196) if deint_196 else ([], False)
        )

        crc_ok = False
        if corrected_ok and len(corrected_bits) == 96:
            payload = corrected_bits[:80]
            rx_crc = bits_to_int(corrected_bits[80:96])
            calc_crc = _crc16_ccitt_dmr(payload)
            # CSBK is commonly CRC-masked by 0xA5A5.
            crc_ok = (calc_crc == rx_crc) or ((calc_crc ^ 0xA5A5) == rx_crc)

        if corrected_ok and crc_ok:
            parsed = _parse_csbk(corrected_bits)

            # Treat trusted CSBK as transmission header context for following voice bursts.
            self._tx_counter += 1
            self._active_tx_meta = {
                "Tx": str(self._tx_counter),
                "PI": "Unknown",
            }

            header = [
                ("Sync", sync_name),
                ("Tx", str(self._tx_counter)),
                ("Decode", "BPTC 196/96 + Hamming(13,9) + CRC16"),
            ] + parsed
            kind = FrameKind.HEADER
        else:
            header = [
                ("Sync", sync_name),
                ("Data", "Decode untrusted"),
                ("FEC", "Column Hamming(13,9) applied" if corrected_ok else "FEC failed"),
                ("CRC", "Pass" if crc_ok else "Fail/Unknown"),
                ("PI", "Unknown"),
            ]
            kind = FrameKind.DATA

        raw_header_bits = (
            burst[SYNC_OFFSET:SYNC_OFFSET + SYNC_LEN]
            + (
                corrected_bits
                if corrected_bits
                else (data_bits if data_bits else data_196)
            )
        )

        return DecodedFrame(
            protocol="DMR",
            kind=kind,
            header_fields=header,
            mbe_frames=[],
            raw_header_bits=raw_header_bits,
            raw_bits=burst,
        )
