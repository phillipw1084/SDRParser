"""
Audio input sources compatible with SDR++ network_sink output.

SDR++ network_sink wire format
-------------------------------
* **TCP mode** – SDR++ is the *TCP server* (listener).  This module acts as
  a TCP *client* that connects to it.
* **UDP mode** – SDR++ is the *UDP sender*; this module *listens* on a local
  port for the incoming datagrams.
* Sample encoding: signed 16-bit integers, little-endian, **no** framing or
  length headers — raw PCM only.
* Default sample rate: 48 000 Hz, mono.
* Typical UDP packet: 512 samples × 2 bytes = 1 024 bytes (set by the
  packer block inside SDR++: ``packer.init(..., 512)``).
"""

from __future__ import annotations

import io
import queue
import socket
import struct
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7355          # SDR++ network_sink default port
DEFAULT_SAMPLE_RATE = 48000  # Hz
CHUNK_SAMPLES = 1024         # samples to yield per iteration


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AudioSource(ABC):
    """Abstract base for all audio input sources.

    Sub-classes must implement :meth:`read_samples` which yields NumPy
    arrays of float32 samples normalised to [-1.0, 1.0].
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE,
                 stereo: bool = False) -> None:
        self.sample_rate = sample_rate
        self.stereo = stereo
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @abstractmethod
    def read_samples(self) -> Generator[np.ndarray, None, None]:
        """Yield chunks of float32 samples normalised to [-1, 1]."""
        ...

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _int16_to_float(raw: bytes) -> np.ndarray:
        """Convert raw SDR++ PCM bytes (int16 LE) to float32 in [-1, 1]."""
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        samples /= 32768.0
        return samples


class BitStreamSource(ABC):
    """Abstract base for bitstream sources.

    Sub-classes yield lists of binary bits (0/1) that are already demodulated
    from RF symbols.
    """

    def __init__(self) -> None:
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @abstractmethod
    def read_bits(self) -> Generator[List[int], None, None]:
        """Yield chunks of bits (0/1)."""
        ...


class TCPBitstreamSource(BitStreamSource):
    """Receive pre-demodulated bit or dibit streams over TCP.

    Supported wire formats
    ----------------------
    * ``auto``: infer from payload
    * ``ascii-bits``: bytes containing ASCII ``0``/``1`` (whitespace ignored)
    * ``dibit-bytes``: one dibit per byte, values 0..3
    * ``packed-dibits``: four dibits packed per byte, MSB dibit first
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        wire_format: str = "auto",
        recv_bytes: int = 4096,
        reconnect_delay: float = 1.0,
    ) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.wire_format = wire_format
        self.recv_bytes = recv_bytes
        self.reconnect_delay = reconnect_delay
        self._sock: Optional[socket.socket] = None

    def stop(self) -> None:
        super().stop()
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _connect(self) -> bool:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((self.host, self.port))
            sock.settimeout(None)
            self._sock = sock
            return True
        except OSError:
            return False

    @staticmethod
    def _dibit_to_bits(dibit: int) -> List[int]:
        return [((dibit >> 1) & 1), (dibit & 1)]

    def _decode_bytes(self, raw: bytes) -> List[int]:
        if not raw:
            return []

        fmt = self.wire_format
        if fmt == "auto":
            ascii_set = {9, 10, 13, 32, 48, 49}
            if all(b in ascii_set for b in raw):
                fmt = "ascii-bits"
            elif all(b <= 3 for b in raw):
                fmt = "dibit-bytes"
            else:
                fmt = "packed-dibits"

        out: List[int] = []
        if fmt == "ascii-bits":
            for b in raw:
                if b == 48:
                    out.append(0)
                elif b == 49:
                    out.append(1)
        elif fmt == "dibit-bytes":
            for b in raw:
                out.extend(self._dibit_to_bits(b & 0x3))
        elif fmt == "packed-dibits":
            for b in raw:
                out.extend(self._dibit_to_bits((b >> 6) & 0x3))
                out.extend(self._dibit_to_bits((b >> 4) & 0x3))
                out.extend(self._dibit_to_bits((b >> 2) & 0x3))
                out.extend(self._dibit_to_bits(b & 0x3))
        else:
            raise ValueError(f"Unsupported TCP bitstream wire format: {fmt}")

        return out

    def read_bits(self) -> Generator[List[int], None, None]:
        while self._running:
            if self._sock is None and not self._connect():
                time.sleep(self.reconnect_delay)
                continue

            try:
                raw = self._sock.recv(self.recv_bytes)
            except OSError:
                raw = b""

            if not raw:
                if self._sock:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                self._sock = None
                time.sleep(self.reconnect_delay)
                continue

            bits = self._decode_bytes(raw)
            if bits:
                yield bits


# ---------------------------------------------------------------------------
# TCP client source
# ---------------------------------------------------------------------------

class TCPClientSource(AudioSource):
    """Connect to SDR++'s TCP network_sink server and stream audio.

    In SDR++, when *TCP* is selected in the network_sink, SDR++ calls
    ``net::listen(hostname, port)`` and waits for a client to connect.
    This class is that client.

    Parameters
    ----------
    host:
        IP / hostname where SDR++ is listening (default ``"127.0.0.1"``).
    port:
        TCP port SDR++ is listening on (default ``7355``).
    sample_rate:
        PCM sample rate configured in SDR++ (default 48 000 Hz).
    stereo:
        ``True`` if SDR++ network_sink is configured for stereo output.
    chunk_samples:
        Number of samples to read per blocking ``recv`` call.
    reconnect_delay:
        Seconds to wait before reconnecting after a dropped connection.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        stereo: bool = False,
        chunk_samples: int = CHUNK_SAMPLES,
        reconnect_delay: float = 2.0,
    ) -> None:
        super().__init__(sample_rate, stereo)
        self.host = host
        self.port = port
        self.chunk_samples = chunk_samples
        self.reconnect_delay = reconnect_delay
        self._sock: Optional[socket.socket] = None

    # ------------------------------------------------------------------

    def start(self) -> None:
        super().start()

    def stop(self) -> None:
        super().stop()
        self._close_socket()

    def _close_socket(self) -> None:
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
                self._sock.close()
            except OSError:
                pass
            finally:
                self._sock = None

    def _connect(self) -> bool:
        """Try to open a TCP connection to SDR++.  Returns True on success."""
        self._close_socket()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((self.host, self.port))
            sock.settimeout(None)
            self._sock = sock
            return True
        except OSError:
            return False

    def _recv_exact(self, n_bytes: int) -> Optional[bytes]:
        """Receive exactly *n_bytes* from the socket, returning None on error."""
        buf = bytearray()
        while len(buf) < n_bytes:
            try:
                chunk = self._sock.recv(n_bytes - len(buf))
            except OSError:
                return None
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def read_samples(self) -> Generator[np.ndarray, None, None]:
        channels = 2 if self.stereo else 1
        n_bytes = self.chunk_samples * channels * 2  # int16 = 2 bytes

        while self._running:
            if self._sock is None:
                if not self._connect():
                    time.sleep(self.reconnect_delay)
                    continue

            raw = self._recv_exact(n_bytes)
            if raw is None:
                self._close_socket()
                time.sleep(self.reconnect_delay)
                continue

            samples = self._int16_to_float(raw)
            if self.stereo:
                # Interleaved L/R → use only the left channel for demodulation
                samples = samples[0::2]
            yield samples


# ---------------------------------------------------------------------------
# UDP listener source
# ---------------------------------------------------------------------------

class UDPSource(AudioSource):
    """Listen on a local UDP port for audio datagrams sent by SDR++.

    In SDR++, when *UDP* is selected in the network_sink, SDR++ calls
    ``net::openUDP("0.0.0.0", port, hostname, port, false)`` and sends
    datagrams to the configured remote host/port.  This class binds a
    UDP socket on ``bind_host:port`` so those datagrams arrive here.

    Parameters
    ----------
    bind_host:
        Local address to bind on (``"0.0.0.0"`` to accept from any
        interface).
    port:
        Local UDP port to bind (default ``7355``).
    sample_rate:
        PCM sample rate configured in SDR++ (default 48 000 Hz).
    stereo:
        ``True`` if the network_sink is configured for stereo output.
    max_queue:
        Maximum number of raw-bytes packets to buffer internally.
    """

    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        stereo: bool = False,
        max_queue: int = 128,
    ) -> None:
        super().__init__(sample_rate, stereo)
        self.bind_host = bind_host
        self.port = port
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=max_queue)
        self._sock: Optional[socket.socket] = None
        self._recv_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------

    def start(self) -> None:
        super().start()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.bind_host, self.port))
        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="udp-recv"
        )
        self._recv_thread.start()

    def stop(self) -> None:
        super().stop()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _recv_loop(self) -> None:
        while self._running and self._sock:
            try:
                data, _ = self._sock.recvfrom(65535)
                if data:
                    try:
                        self._queue.put_nowait(data)
                    except queue.Full:
                        pass  # drop oldest if consumer is slow
            except OSError:
                break

    def read_samples(self) -> Generator[np.ndarray, None, None]:
        while self._running:
            try:
                raw = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            samples = self._int16_to_float(raw)
            if self.stereo and len(samples) % 2 == 0:
                samples = samples[0::2]  # left channel only
            yield samples


# ---------------------------------------------------------------------------
# File / WAV source (for testing and off-line analysis)
# ---------------------------------------------------------------------------

class FileSource(AudioSource):
    """Read raw signed-16-bit PCM or WAV audio from a file.

    Parameters
    ----------
    path:
        Path to the input file.  If the file has a ``.wav`` extension the
        header is parsed automatically; otherwise it is treated as raw
        16-bit little-endian mono PCM.
    sample_rate:
        Sample rate to report (for raw files this must be set correctly;
        for WAV files the value embedded in the header is used instead).
    chunk_samples:
        Samples to emit per iteration.
    loop:
        If ``True``, loop back to the start when the file is exhausted.
    """

    def __init__(
        self,
        path: str | Path,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_samples: int = CHUNK_SAMPLES,
        loop: bool = False,
    ) -> None:
        super().__init__(sample_rate, stereo=False)
        self.path = Path(path)
        self.chunk_samples = chunk_samples
        self.loop = loop

    def read_samples(self) -> Generator[np.ndarray, None, None]:
        while True:
            if self.path.suffix.lower() == ".wav":
                yield from self._read_wav()
            else:
                yield from self._read_raw()
            if not self.loop or not self._running:
                break

    def _read_raw(self) -> Generator[np.ndarray, None, None]:
        n_bytes = self.chunk_samples * 2
        with open(self.path, "rb") as fh:
            while self._running:
                raw = fh.read(n_bytes)
                if len(raw) < 2:
                    break
                # Pad if last chunk is short
                if len(raw) < n_bytes:
                    raw = raw + b"\x00" * (n_bytes - len(raw))
                yield self._int16_to_float(raw)

    def _read_wav(self) -> Generator[np.ndarray, None, None]:
        """Minimal WAV parser — handles PCM (fmt chunk type 1) only."""
        with open(self.path, "rb") as fh:
            header = fh.read(44)
            if len(header) < 44 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise ValueError(f"Not a valid WAV file: {self.path}")

            num_channels = struct.unpack_from("<H", header, 22)[0]
            sample_rate = struct.unpack_from("<I", header, 24)[0]
            bits_per_sample = struct.unpack_from("<H", header, 34)[0]

            self.sample_rate = sample_rate

            # Skip to data chunk (simplistic — assumes standard 44-byte header)
            n_bytes = self.chunk_samples * num_channels * (bits_per_sample // 8)
            while self._running:
                raw = fh.read(n_bytes)
                if len(raw) < (bits_per_sample // 8) * num_channels:
                    break
                if bits_per_sample == 16:
                    samples = self._int16_to_float(raw)
                elif bits_per_sample == 8:
                    raw_u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                    samples = (raw_u8 - 128.0) / 128.0
                else:
                    raise ValueError(f"Unsupported bit depth: {bits_per_sample}")

                if num_channels > 1:
                    # Take left channel only
                    samples = samples[0::num_channels]
                yield samples
