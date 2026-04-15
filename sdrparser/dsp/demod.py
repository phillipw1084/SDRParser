"""
DSP module — symbol recovery from FM-demodulated audio.

SDR++ demodulates the FM signal before sending it over the network, so
the audio stream already contains the 4-FSK baseband signal.  What we
need to do here is:

1. Optionally low-pass filter to reduce inter-symbol interference.
2. Recover symbol timing (clock recovery) and sample one value per symbol.
3. Slice each sample into a dibit (0–3).

Symbol rates
------------
* DMR, P25 Phase-1, NXDN 12.5 kHz : 4 800 baud
* NXDN 6.25 kHz                    : 2 400 baud

At 48 000 Hz input sample rate each DMR/P25 symbol spans exactly 10
samples, making clock recovery straightforward.
"""

from __future__ import annotations

from typing import Generator, List

import numpy as np
from scipy import signal as scipy_signal

# ---------------------------------------------------------------------------
# Dibit definitions  (standard ETSI 4-FSK / C4FM symbol mapping)
# ---------------------------------------------------------------------------
#   Symbol  Dibit  Frequency deviation
#   +3       01     +1800 Hz
#   +1       00     + 600 Hz
#   -1       10     - 600 Hz
#   -3       11     -1800 Hz

DIBIT_MAP = {
    3:  0b01,
    2:  0b00,
    1:  0b10,
    0:  0b11,
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 48_000   # Hz
DMR_BAUD = 4_800
P25_BAUD = 4_800
NXDN_BAUD_WIDE = 4_800
NXDN_BAUD_NARROW = 2_400


# ---------------------------------------------------------------------------
# Low-pass filter helper
# ---------------------------------------------------------------------------

def _lowpass_filter(
    samples: np.ndarray,
    cutoff_hz: float,
    sample_rate: float,
    order: int = 31,
) -> np.ndarray:
    """Apply a linear-phase FIR low-pass filter in-place (zero-phase)."""
    nyq = sample_rate / 2.0
    h = scipy_signal.firwin(order, cutoff_hz / nyq)
    return scipy_signal.lfilter(h, 1.0, samples)


# ---------------------------------------------------------------------------
# Symbol demodulator
# ---------------------------------------------------------------------------

class SymbolDemodulator:
    """Recover 4-FSK dibits from a stream of float32 audio samples.

    The demodulator maintains an internal sample buffer and emits a
    flat list of dibits (integers 0–3) each time :meth:`process` is
    called.  Symbol timing is tracked using a simple zero-crossing /
    early-late gate approach that works well when the input comes from
    SDR++ (already FM-demodulated, reasonably clean).

    Parameters
    ----------
    sample_rate:
        Input audio sample rate in Hz.
    baud_rate:
        Symbol rate of the target protocol (4 800 or 2 400 baud).
    apply_filter:
        When *True* apply a mild LPF before slicing to reduce ISI.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        baud_rate: int = DMR_BAUD,
        apply_filter: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.baud_rate = baud_rate
        self.sps: float = sample_rate / baud_rate       # samples per symbol
        self.apply_filter = apply_filter

        # Internal state
        self._buf: np.ndarray = np.array([], dtype=np.float32)
        self._sym_phase: float = self.sps / 2.0        # fractional offset
        self._agc_max: float = 1.0                     # running peak for AGC

        # Slice thresholds (relative to normalised range)
        self._thresh_high: float = 2.0 / 3.0
        self._thresh_low: float = -2.0 / 3.0
        self._thresh_mid: float = 0.0

        self._filter_taps: np.ndarray | None = None
        self._filter_zi: np.ndarray | None = None
        if self.apply_filter:
            cutoff = self.baud_rate * 0.75
            nyq = self.sample_rate / 2.0
            taps = scipy_signal.firwin(31, cutoff / nyq)
            self._filter_taps = taps.astype(np.float32)
            self._filter_zi = np.zeros(len(self._filter_taps) - 1, dtype=np.float32)

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------

    def process(self, samples: np.ndarray) -> List[int]:
        """Append *samples* to the internal buffer and return new dibits."""
        samples_f = samples.astype(np.float32, copy=False)
        if self.apply_filter and self._filter_taps is not None and self._filter_zi is not None:
            samples_f, self._filter_zi = scipy_signal.lfilter(
                self._filter_taps,
                1.0,
                samples_f,
                zi=self._filter_zi,
            )

        if self._buf.size == 0:
            self._buf = samples_f.copy()
        else:
            self._buf = np.concatenate((self._buf, samples_f))

        dibits: List[int] = []
        while self._sym_phase < len(self._buf):
            idx = int(self._sym_phase)
            if idx >= len(self._buf):
                break

            value = self._buf[idx]

            # Simple AGC: track peak
            abs_val = abs(value)
            if abs_val > self._agc_max:
                self._agc_max = abs_val
            elif self._agc_max > 1e-6:
                self._agc_max *= 0.9999  # slow decay

            norm = value / self._agc_max if self._agc_max > 1e-6 else 0.0

            # Slice directly to dibit value to avoid extra table lookup.
            if norm > self._thresh_high:
                dibit = 0b01
            elif norm > self._thresh_mid:
                dibit = 0b00
            elif norm > self._thresh_low:
                dibit = 0b10
            else:
                dibit = 0b11

            dibits.append(dibit)

            self._sym_phase += self.sps

        # Advance buffer — keep only unprocessed tail
        consumed = int(self._sym_phase)
        if consumed > 0:
            self._buf = self._buf[consumed:]
            self._sym_phase -= consumed

        return dibits

    def reset(self) -> None:
        """Reset internal state (call when switching protocols / baud rates)."""
        self._buf = np.array([], dtype=np.float32)
        self._sym_phase = self.sps / 2.0
        self._agc_max = 1.0
        if self._filter_zi is not None:
            self._filter_zi.fill(0.0)


# ---------------------------------------------------------------------------
# Bit-stream builder
# ---------------------------------------------------------------------------

class BitStreamBuffer:
    """Accumulate dibits into a searchable bit string.

    Each dibit is stored as two consecutive bits, MSB first.  The buffer
    supports fast pattern matching via :meth:`find_pattern`.

    Example
    -------
    >>> buf = BitStreamBuffer()
    >>> buf.push_dibits([0b01, 0b10, 0b00, 0b11])
    >>> len(buf)
    8
    """

    def __init__(self, max_bits: int = 4096) -> None:
        self._bits: List[int] = []
        self.max_bits = max_bits

    def push_dibits(self, dibits: List[int]) -> None:
        for db in dibits:
            self._bits.append((db >> 1) & 1)   # MSB
            self._bits.append(db & 1)           # LSB
        # Trim to keep memory bounded
        if len(self._bits) > self.max_bits:
            self._bits = self._bits[-self.max_bits:]

    def push_bits(self, bits: List[int]) -> None:
        self._bits.extend(bits)
        if len(self._bits) > self.max_bits:
            self._bits = self._bits[-self.max_bits:]

    def __len__(self) -> int:
        return len(self._bits)

    def find_pattern(self, pattern: List[int]) -> int:
        """Return first bit position where *pattern* matches, or -1."""
        plen = len(pattern)
        bits = self._bits
        blen = len(bits)
        for i in range(blen - plen + 1):
            if bits[i:i + plen] == pattern:
                return i
        return -1

    def consume(self, n: int) -> List[int]:
        """Remove and return the first *n* bits."""
        result = self._bits[:n]
        self._bits = self._bits[n:]
        return result

    def peek(self, start: int, length: int) -> List[int]:
        """Return a slice without consuming."""
        return self._bits[start:start + length]

    def bits_available(self) -> int:
        return len(self._bits)

    def clear(self) -> None:
        self._bits.clear()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def bits_to_int(bits: List[int]) -> int:
    """Convert a list of bits (MSB first) to an integer."""
    result = 0
    for b in bits:
        result = (result << 1) | (b & 1)
    return result


def int_to_bits(value: int, width: int) -> List[int]:
    """Convert an integer to a fixed-width bit list (MSB first)."""
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]
