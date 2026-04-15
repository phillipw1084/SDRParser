"""
Base classes for digital voice protocol decoders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from sdrparser.mbe.frames import MBEFrame


# ---------------------------------------------------------------------------
# Common enumerations
# ---------------------------------------------------------------------------

class FrameKind(Enum):
    VOICE   = auto()
    DATA    = auto()
    HEADER  = auto()
    CONTROL = auto()
    UNKNOWN = auto()


# ---------------------------------------------------------------------------
# Decoded frame dataclass
# ---------------------------------------------------------------------------

@dataclass
class DecodedFrame:
    """Represents one fully decoded protocol frame.

    Attributes
    ----------
    protocol:
        String name: ``"DMR"``, ``"P25"``, or ``"NXDN"``.
    kind:
        High-level frame type.
    header_fields:
        Ordered list of ``(label, value)`` tuples for GUI display.
    mbe_frames:
        Zero, one, or two MBE codec frames carried in this burst.
    raw_bits:
        The complete frame as a bit list (useful for debugging).
    """

    protocol: str
    kind: FrameKind
    header_fields: List[tuple[str, str]] = field(default_factory=list)
    mbe_frames: List[MBEFrame] = field(default_factory=list)
    raw_bits: List[int] = field(default_factory=list)

    def summary(self) -> str:
        """One-line human-readable summary of this frame."""
        fields = ", ".join(f"{k}={v}" for k, v in self.header_fields)
        return f"[{self.protocol}] {self.kind.name} | {fields}"


# ---------------------------------------------------------------------------
# Base decoder
# ---------------------------------------------------------------------------

class ProtocolDecoder:
    """Abstract base class for protocol-specific decoders.

    Sub-classes implement :meth:`push_bits` which accepts a list of bits
    from the bit-stream buffer and returns any :class:`DecodedFrame`
    objects that can be assembled.

    Each decoder keeps a *sync_count* (number of frames decoded in the
    current run) and a *sync_lost* flag to indicate when frame alignment
    has been lost.
    """

    PROTOCOL_NAME: str = "UNKNOWN"

    def __init__(self) -> None:
        self.sync_count: int = 0
        self.sync_lost: bool = True
        self._frame_index: int = 0

    # ------------------------------------------------------------------
    # Sub-class interface
    # ------------------------------------------------------------------

    def push_bits(self, bits: List[int]) -> List[DecodedFrame]:
        """Process incoming bits and return any complete frames found.

        Sub-classes must override this method.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset decoder state (e.g. after silence / protocol switch)."""
        self.sync_count = 0
        self.sync_lost = True
        self._frame_index = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bits_to_int(bits: List[int]) -> int:
        result = 0
        for b in bits:
            result = (result << 1) | (b & 1)
        return result

    @staticmethod
    def _hamming_distance(a: List[int], b: List[int]) -> int:
        return sum(x != y for x, y in zip(a, b))
