"""
MBE (Multi-Band Excitation) codec frame handling.

This module provides interleaving and deinterleaving for the three MBE
codec variants used by the supported digital voice protocols:

* **AMBE+2** (72-bit frames) — used by DMR and NXDN
* **IMBE**   (88-bit frames) — used by P25 Phase 1

Terminology
-----------
*Interleaved*
    Bits as they appear in the **transmitted** radio burst.  Consecutive
    codec bits are deliberately spread across the frame to improve
    resilience against burst errors.

*Deinterleaved*
    Bits in the natural order expected by the **codec** (DVSI AMBE+2 /
    IMBE chip or compatible software).

References
----------
* DMR  interleaving: ETSI TS 102 361-1 §B.2
* P25  interleaving: TIA-102.BABB (IMBE vocoder standard)
* NXDN interleaving: ETSI TS 102 166-2 §5.3
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

# ---------------------------------------------------------------------------
# Frame type enum
# ---------------------------------------------------------------------------

class MBEType(Enum):
    AMBE2 = auto()   # 72-bit, used by DMR and NXDN
    IMBE  = auto()   # 88-bit, used by P25 Phase 1


# ---------------------------------------------------------------------------
# Interleave tables
# ---------------------------------------------------------------------------
#
# Each table maps: interleaved_position → deinterleaved_position
#
# To *deinterleave* an array ``interleaved[i]``:
#     deinterleaved[table[i]] = interleaved[i]
#
# To *interleave* an array ``deinterleaved[i]``:
#     interleaved[i] = deinterleaved[table[i]]

# DMR AMBE+2 — 72-bit frame
# Diagonal interleave across a 4-row × 18-column matrix
# (ETSI TS 102 361-1, Table B.6 / B.7)
_DMR_AMBE2_TABLE: List[int] = (
    # row 0: positions 0, 4, 8, …, 68
    [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68] +
    # row 1: positions 1, 5, 9, …, 69
    [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69] +
    # row 2: positions 2, 6, 10, …, 70
    [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70] +
    # row 3: positions 3, 7, 11, …, 71
    [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71]
)  # length = 72 ✓

# NXDN AMBE+2 — 72-bit frame
# Column-major interleave across a 3-row × 24-column matrix
# (ETSI TS 102 166-2 §5.3.3)
_NXDN_AMBE2_TABLE: List[int] = (
    [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69] +
    [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70] +
    [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71]
)  # length = 72 ✓

# P25 IMBE — 88-bit frame
# Column-major interleave across a 6-row matrix
# (TIA-102.BABB Table 8)
_P25_IMBE_TABLE: List[int] = (
    # 6 rows × varying widths = 88 bits
    [0,  7, 12, 19, 24, 31, 36, 43, 48, 55, 60, 67, 72, 79, 84] +  # row 0 (15)
    [1,  6, 13, 18, 25, 30, 37, 42, 49, 54, 61, 66, 73, 78, 85] +  # row 1 (15)
    [2,  5, 14, 17, 26, 29, 38, 41, 50, 53, 62, 65, 74, 77, 86] +  # row 2 (15)
    [3,  4, 15, 16, 27, 28, 39, 40, 51, 52, 63, 64, 75, 76, 87] +  # row 3 (15)
    [8, 11, 20, 23, 32, 35, 44, 47, 56, 59, 68, 71, 80, 83]      + # row 4 (14)
    [9, 10, 21, 22, 33, 34, 45, 46, 57, 58, 69, 70, 81, 82]        # row 5 (14)
)  # length = 88 ✓

# Expose as a dict for easy lookup by protocol
INTERLEAVE_TABLES = {
    "DMR":  _DMR_AMBE2_TABLE,
    "NXDN": _NXDN_AMBE2_TABLE,
    "P25":  _P25_IMBE_TABLE,
}

FRAME_BITS = {
    MBEType.AMBE2: 72,
    MBEType.IMBE:  88,
}


# DMR AMBE bit placement schedule (mirrors dsd-fme dmr_const.h rW/rX/rY/rZ).
_DMR_RW: List[int] = [
    0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 2,
    0, 2, 0, 2, 0, 2,
    0, 2, 0, 2, 0, 2,
]
_DMR_RX: List[int] = [
    23, 10, 22, 9, 21, 8,
    20, 7, 19, 6, 18, 5,
    17, 4, 16, 3, 15, 2,
    14, 1, 13, 0, 12, 10,
    11, 9, 10, 8, 9, 7,
    8, 6, 7, 5, 6, 4,
]
_DMR_RY: List[int] = [
    0, 2, 0, 2, 0, 2,
    0, 2, 0, 3, 0, 3,
    1, 3, 1, 3, 1, 3,
    1, 3, 1, 3, 1, 3,
    1, 3, 1, 3, 1, 3,
    1, 3, 1, 3, 1, 3,
]
_DMR_RZ: List[int] = [
    5, 3, 4, 2, 3, 1,
    2, 0, 1, 13, 0, 12,
    22, 11, 21, 10, 20, 9,
    19, 8, 18, 7, 17, 6,
    16, 5, 15, 4, 14, 3,
    13, 2, 12, 1, 11, 0,
]


def _dmr_deinterleave_ambe72(interleaved: List[int]) -> List[int]:
    """Convert 72 interleaved DMR bits into AMBE row-segment order.

    Output order matches AMBE rows used by dsd-fme codeword handling:
    row0[24] + row1[23] + row2[11] + row3[14] = 72 bits.
    """
    if len(interleaved) != 72:
        raise ValueError(f"DMR interleaved length {len(interleaved)} != 72")

    ambe = [[0] * 24 for _ in range(4)]
    for i in range(36):
        ambe[_DMR_RW[i]][_DMR_RX[i]] = interleaved[(i * 2) + 0] & 1
        ambe[_DMR_RY[i]][_DMR_RZ[i]] = interleaved[(i * 2) + 1] & 1

    return ambe[0][0:24] + ambe[1][0:23] + ambe[2][0:11] + ambe[3][0:14]


def _dmr_interleave_ambe72(deinterleaved: List[int]) -> List[int]:
    """Inverse of _dmr_deinterleave_ambe72."""
    if len(deinterleaved) != 72:
        raise ValueError(f"DMR deinterleaved length {len(deinterleaved)} != 72")

    ambe = [[0] * 24 for _ in range(4)]
    k = 0
    for c in range(24):
        ambe[0][c] = deinterleaved[k] & 1
        k += 1
    for c in range(23):
        ambe[1][c] = deinterleaved[k] & 1
        k += 1
    for c in range(11):
        ambe[2][c] = deinterleaved[k] & 1
        k += 1
    for c in range(14):
        ambe[3][c] = deinterleaved[k] & 1
        k += 1

    interleaved = [0] * 72
    for i in range(36):
        interleaved[(i * 2) + 0] = ambe[_DMR_RW[i]][_DMR_RX[i]]
        interleaved[(i * 2) + 1] = ambe[_DMR_RY[i]][_DMR_RZ[i]]
    return interleaved


# ---------------------------------------------------------------------------
# Core interleave / deinterleave functions
# ---------------------------------------------------------------------------

def deinterleave(interleaved: List[int], table: List[int]) -> List[int]:
    """Reorder *interleaved* bits into codec-natural order using *table*.

    ``table[i]`` is the destination position for the bit at position ``i``
    in the interleaved sequence.

    Parameters
    ----------
    interleaved:
        Bits as received from the radio frame (length must equal
        ``len(table)``).
    table:
        Protocol-specific deinterleave index table.

    Returns
    -------
    List[int]
        Bits in codec-input order.
    """
    n = len(table)
    if len(interleaved) != n:
        raise ValueError(
            f"interleaved length {len(interleaved)} != table length {n}"
        )
    result = [0] * n
    for src, dst in enumerate(table):
        result[dst] = interleaved[src]
    return result


def interleave(deinterleaved: List[int], table: List[int]) -> List[int]:
    """Perform the inverse of :func:`deinterleave`.

    ``table[i]`` is the source position in *deinterleaved* for position
    ``i`` in the output (interleaved) sequence.

    Parameters
    ----------
    deinterleaved:
        Codec bits in natural order.
    table:
        Same protocol-specific table used by :func:`deinterleave`.

    Returns
    -------
    List[int]
        Bits in over-the-air transmission order.
    """
    n = len(table)
    if len(deinterleaved) != n:
        raise ValueError(
            f"deinterleaved length {len(deinterleaved)} != table length {n}"
        )
    result = [0] * n
    for dst, src in enumerate(table):
        result[dst] = deinterleaved[src]
    return result


# ---------------------------------------------------------------------------
# MBEFrame dataclass
# ---------------------------------------------------------------------------

@dataclass
class MBEFrame:
    """A single MBE codec frame with both interleaved and deinterleaved bits.

    Attributes
    ----------
    protocol:
        One of ``"DMR"``, ``"P25"``, ``"NXDN"``.
    frame_type:
        :class:`MBEType` enum value.
    frame_index:
        Sequential frame number within the current call (0-based).
    interleaved_bits:
        Bits exactly as extracted from the over-the-air frame.
    deinterleaved_bits:
        Same bits after deinterleaving (codec-ready order).
    """

    protocol: str
    frame_type: MBEType
    frame_index: int
    interleaved_bits: List[int]
    deinterleaved_bits: List[int]

    # ------------------------------------------------------------------

    @classmethod
    def from_interleaved(
        cls,
        protocol: str,
        frame_type: MBEType,
        frame_index: int,
        bits: List[int],
    ) -> "MBEFrame":
        """Build an MBEFrame from raw over-the-air *bits*.

        The bits are deinterleaved automatically using the correct table
        for *protocol*.
        """
        if protocol == "DMR" and frame_type == MBEType.AMBE2:
            deinterleaved = _dmr_deinterleave_ambe72(bits)
        else:
            table = INTERLEAVE_TABLES[protocol]
            deinterleaved = deinterleave(bits, table)
        return cls(
            protocol=protocol,
            frame_type=frame_type,
            frame_index=frame_index,
            interleaved_bits=list(bits),
            deinterleaved_bits=deinterleaved,
        )

    @classmethod
    def from_deinterleaved(
        cls,
        protocol: str,
        frame_type: MBEType,
        frame_index: int,
        bits: List[int],
    ) -> "MBEFrame":
        """Build an MBEFrame from codec-order *bits* (e.g., for test injection)."""
        if protocol == "DMR" and frame_type == MBEType.AMBE2:
            interleaved_bits = _dmr_interleave_ambe72(bits)
        else:
            table = INTERLEAVE_TABLES[protocol]
            interleaved_bits = interleave(bits, table)
        return cls(
            protocol=protocol,
            frame_type=frame_type,
            frame_index=frame_index,
            interleaved_bits=interleaved_bits,
            deinterleaved_bits=list(bits),
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def bits_hex(self, which: str = "deinterleaved") -> str:
        """Return bits as a hexadecimal string.

        Parameters
        ----------
        which:
            ``"interleaved"`` or ``"deinterleaved"`` (default).
        """
        bits = (self.interleaved_bits if which == "interleaved"
                else self.deinterleaved_bits)
        return _bits_to_hex(bits)

    def bits_str(self, which: str = "deinterleaved") -> str:
        """Return bits as a compact binary string (e.g. ``'011010…'``)."""
        bits = (self.interleaved_bits if which == "interleaved"
                else self.deinterleaved_bits)
        return "".join(str(b) for b in bits)

    def bits_hex_compact(self, which: str = "deinterleaved") -> str:
        """Return bits as uppercase HEX without spaces."""
        return self.bits_hex(which).replace(" ", "")

    def ambe_hex_49(self) -> str:
        """Return dsd-fme style AMBE 49-bit HEX (14 hex chars).

        dsd-fme derives this from AMBE rows c0/c1/v2/v3 and left-shifts by 7.
        Only valid for AMBE2 frames (DMR/NXDN).
        """
        if self.frame_type != MBEType.AMBE2 or len(self.deinterleaved_bits) != 72:
            return ""

        fr0 = self.deinterleaved_bits[0:24]
        fr1 = self.deinterleaved_bits[24:47]
        fr2 = self.deinterleaved_bits[47:58]
        fr3 = self.deinterleaved_bits[58:72]

        c0 = _bits_to_int(fr0[::-1][:12])
        c1 = _bits_to_int(fr1[::-1][:12])
        v2 = _bits_to_int(fr2)
        v3 = _bits_to_int(fr3)

        hex49 = (
            ((c0 & 0xFFF) << 37)
            | ((c1 & 0xFFF) << 25)
            | ((v2 & 0x7FF) << 14)
            | (v3 & 0x3FFF)
        )
        return f"{(hex49 << 7):014X}"

    def ambe_hex_49_short(self) -> str:
        """Return 12-hex shortened AMBE display (drop final pad byte)."""
        full = self.ambe_hex_49()
        return full[:-2] if len(full) == 14 else full

    def __repr__(self) -> str:
        return (
            f"MBEFrame(protocol={self.protocol!r}, type={self.frame_type.name}, "
            f"index={self.frame_index}, "
            f"interleaved={self.bits_hex('interleaved')}, "
            f"deinterleaved={self.bits_hex('deinterleaved')})"
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _bits_to_hex(bits: List[int]) -> str:
    """Pack a bit list into a hexadecimal string (zero-padded to full bytes)."""
    # Pad to full byte boundary
    pad = (-len(bits)) % 8
    padded = bits + [0] * pad
    result = []
    for i in range(0, len(padded), 8):
        byte_val = 0
        for b in padded[i:i + 8]:
            byte_val = (byte_val << 1) | (b & 1)
        result.append(f"{byte_val:02X}")
    return " ".join(result)


def bits_to_hex_compact(bits: List[int]) -> str:
    """Pack bits into uppercase HEX without spaces (dsd-fme style)."""
    return _bits_to_hex(bits).replace(" ", "")


def _bits_to_int(bits: List[int]) -> int:
    """Convert MSB-first bits to int."""
    out = 0
    for b in bits:
        out = (out << 1) | (b & 1)
    return out
