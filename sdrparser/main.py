"""
SDRParser main processing pipeline.

Wires together audio input → symbol demodulation → multi-protocol
detection → frame output.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, List, Optional

from sdrparser.audio.input import AudioSource, TCPClientSource, UDPSource
from sdrparser.dsp.demod import SymbolDemodulator, BitStreamBuffer
from sdrparser.protocols.base import DecodedFrame
from sdrparser.protocols.dmr import DMRDecoder
from sdrparser.protocols.nxdn import NXDNDecoder
from sdrparser.protocols.p25 import P25Decoder

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

FrameCallback = Callable[[DecodedFrame], None]


class SDRParserPipeline:
    """End-to-end processing pipeline.

    Parameters
    ----------
    source:
        An :class:`~sdrparser.audio.input.AudioSource` instance.
    on_frame:
        Callable invoked (from the worker thread) for every decoded frame.
    baud_rate:
        Expected symbol rate.  Use 4800 for DMR/P25/NXDN-96 or 2400 for
        NXDN-48.
    enabled_protocols:
        Which decoders to run.  Defaults to all three.
    """

    def __init__(
        self,
        source: AudioSource,
        on_frame: FrameCallback,
        baud_rate: int = 4800,
        enabled_protocols: Optional[List[str]] = None,
    ) -> None:
        self.source = source
        self.on_frame = on_frame

        if enabled_protocols is None:
            enabled_protocols = ["DMR", "P25", "NXDN"]

        self._demod = SymbolDemodulator(
            sample_rate=source.sample_rate,
            baud_rate=baud_rate,
        )
        self._bit_buf = BitStreamBuffer(max_bits=8192)

        self._decoders: List = []
        if "DMR" in enabled_protocols:
            self._decoders.append(DMRDecoder())
        if "P25" in enabled_protocols:
            self._decoders.append(P25Decoder())
        if "NXDN" in enabled_protocols:
            self._decoders.append(NXDNDecoder())

        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.source.start()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="sdrparser-pipeline"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self.source.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        for samples in self.source.read_samples():
            if not self._running:
                break
            dibits = self._demod.process(samples)
            if not dibits:
                continue
            self._bit_buf.push_dibits(dibits)
            bits = self._bit_buf.consume(self._bit_buf.bits_available())
            for decoder in self._decoders:
                frames = decoder.push_bits(list(bits))
                for frame in frames:
                    try:
                        self.on_frame(frame)
                    except Exception:
                        pass
