#!/usr/bin/env python3
"""
run.py — SDRParser entry point.

Usage examples
--------------
# Launch the GUI (default)
python run.py

# Decode a WAV file in headless mode
python run.py --source file --file /path/to/recording.wav

# Connect to SDR++ TCP network_sink (SDR++ listens on 127.0.0.1:7355)
python run.py --source tcp --host 127.0.0.1 --port 7355

# Listen for SDR++ UDP network_sink datagrams
python run.py --source udp --port 7355

# Restrict to P25 only
python run.py --protocols P25

# Override baud rate (NXDN 6.25 kHz = 2400 baud)
python run.py --baud 2400 --protocols NXDN
"""

import argparse
import shlex
import signal
import sys
import time
from typing import List

from sdrparser.audio.input import (
    DEFAULT_PORT,
    DEFAULT_SAMPLE_RATE,
    FileSource,
    TCPBitstreamSource,
    TCPClientSource,
    UDPSource,
)
from sdrparser.main import SDRParserPipeline, BitstreamParserPipeline
from sdrparser.backends.dsdfme import DsdfmeCommandError, stream_dsdfme_events
from sdrparser.protocols.base import DecodedFrame


def _derive_pi_status(frame: DecodedFrame) -> str:
    """Derive a readable privacy/encryption indicator from header fields."""
    fields = dict(frame.header_fields)
    algid = fields.get("AlgID", "")
    svcopt = fields.get("SvcOpt", "")

    if algid:
        if "No encryption" in algid:
            return "Clear"
        return f"Encrypted ({algid})"

    if "Privacy=On" in svcopt:
        return "Protected"
    if "Privacy=Off" in svcopt:
        return "Clear"

    return "Unknown"


def _format_header_lines(frame: DecodedFrame) -> List[str]:
    lines = ["Header"]
    lines.append(f"  PI: {_derive_pi_status(frame)}")
    for key, value in frame.header_fields:
        lines.append(f"  {key}: {value}")

    header_hex = frame.header_hex_compact()
    if header_hex:
        lines.append(f"  Raw HEX: {header_hex}")
    return lines


def _format_vocoder_lines(frame: DecodedFrame, show_bits: bool) -> List[str]:
    lines = ["Vocoder"]
    if not frame.mbe_frames:
        lines.append("  None")
        return lines

    for mf in frame.mbe_frames:
        lines.append(f"  Frame #{mf.frame_index} [{mf.frame_type.name}]")
        inter_hex = mf.bits_hex_compact("interleaved")
        deint_hex = mf.bits_hex_compact("deinterleaved")

        if mf.frame_type.name == "AMBE2":
            ambe49 = mf.ambe_hex_49()
            if ambe49:
                lines.append(f"    AMBE HEX(49): {ambe49}")
                lines.append(f"    AMBE HEX(49-S): {mf.ambe_hex_49_short()}")
            lines.append(f"    AMBE HEX(72) INT: {inter_hex}")
            lines.append(f"    AMBE HEX(72) DEINT: {deint_hex}")
        else:
            lines.append(f"    IMBE HEX(88) INT: {inter_hex}")
            lines.append(f"    IMBE HEX(88) DEINT: {deint_hex}")

        if show_bits:
            lines.append(f"    INT BITS: {mf.bits_str('interleaved')}")
            lines.append(f"    DEINT BITS: {mf.bits_str('deinterleaved')}")

    return lines


def _print_frame(frame: DecodedFrame, show_bits: bool = False) -> None:
    title = f"[{frame.protocol}] {frame.kind.name}"
    print(f"{title}\n{'-' * len(title)}")
    for line in _format_header_lines(frame):
        print(line)
    for line in _format_vocoder_lines(frame, show_bits):
        print(line)
    print()


def _run_headless(args: argparse.Namespace) -> None:
    protos = [p.strip().upper() for p in args.protocols.split(",")]

    if args.source == "tcp":
        source = TCPClientSource(
            host=args.host, port=args.port, sample_rate=args.sample_rate
        )
        pipeline = SDRParserPipeline(
            source=source,
            on_frame=lambda f: _print_frame(f, show_bits=args.show_bits),
            baud_rate=args.baud,
            enabled_protocols=protos,
        )
    elif args.source == "tcp-bits":
        source = TCPBitstreamSource(
            host=args.host,
            port=args.port,
            wire_format=args.bit_format,
        )
        pipeline = BitstreamParserPipeline(
            source=source,
            on_frame=lambda f: _print_frame(f, show_bits=args.show_bits),
            enabled_protocols=protos,
        )
    elif args.source == "udp":
        source = UDPSource(
            bind_host="0.0.0.0", port=args.port, sample_rate=args.sample_rate
        )
        pipeline = SDRParserPipeline(
            source=source,
            on_frame=lambda f: _print_frame(f, show_bits=args.show_bits),
            baud_rate=args.baud,
            enabled_protocols=protos,
        )
    elif args.source == "file":
        if not args.file:
            print("Error: --file is required when --source=file", file=sys.stderr)
            sys.exit(1)
        source = FileSource(path=args.file, sample_rate=args.sample_rate)
        pipeline = SDRParserPipeline(
            source=source,
            on_frame=lambda f: _print_frame(f, show_bits=args.show_bits),
            baud_rate=args.baud,
            enabled_protocols=protos,
        )
    else:
        print(f"Unknown source: {args.source}", file=sys.stderr)
        sys.exit(1)

    stop_event = [False]

    def _handler(sig, frame):
        print("\nStopping…", file=sys.stderr)
        stop_event[0] = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    pipeline.start()
    print(f"SDRParser running — source={args.source}, protocols={protos}",
          file=sys.stderr)

    try:
        while not stop_event[0]:
            time.sleep(0.25)
    finally:
        pipeline.stop()


def _run_dsdfme_backend(args: argparse.Namespace) -> None:
    if not args.dsd_fme_cmd:
        print(
            "Error: --dsd-fme-cmd is required when --backend=dsd-fme",
            file=sys.stderr,
        )
        sys.exit(2)

    cmd = shlex.split(args.dsd_fme_cmd)
    print(f"Running dsd-fme backend: {' '.join(cmd)}", file=sys.stderr)

    tx_counter = 0
    last_header = ""

    try:
        event_stream = stream_dsdfme_events(cmd)
        while True:
            try:
                event = next(event_stream)
            except StopIteration as done:
                code = int(done.value or 0)
                if code != 0:
                    print(f"dsd-fme exited with code {code}", file=sys.stderr)
                    sys.exit(code)
                break

            if event.kind == "header":
                if event.text == last_header:
                    continue
                last_header = event.text
                tx_counter += 1
                print(f"[{event.protocol}] HEADER  Tx={tx_counter}")
                print(f"  {event.text}")
            elif event.kind == "vocoder":
                print(f"[{event.protocol}] VOCODER {event.vocoder_type}: {event.vocoder_hex}")
    except DsdfmeCommandError as exc:
        print(f"Failed to launch dsd-fme backend: {exc}", file=sys.stderr)
        sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SDRParser — DMR/P25/NXDN digital voice decoder for SDR++ audio"
    )
    parser.add_argument(
        "--gui", action="store_true", default=False,
        help="Launch the GUI (default when no --source is given)"
    )
    parser.add_argument(
        "--source", choices=["tcp", "tcp-bits", "udp", "file"], default=None,
        help="Audio input source (omit to launch GUI)"
    )
    parser.add_argument(
        "--bit-format",
        choices=["auto", "ascii-bits", "dibit-bytes", "packed-dibits"],
        default="auto",
        help="Wire format used by --source=tcp-bits",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="SDR++ hostname (TCP mode)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port number (default {DEFAULT_PORT})")
    parser.add_argument("--file", default=None,
                        help="Path to WAV or raw PCM file")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"PCM sample rate (default {DEFAULT_SAMPLE_RATE})")
    parser.add_argument("--baud", type=int, default=4800,
                        help="Symbol rate in baud (4800 for DMR/P25, 2400 for NXDN-6.25k)")
    parser.add_argument("--protocols", default="DMR,P25,NXDN",
                        help="Comma-separated list of protocols to enable")
    parser.add_argument(
        "--show-bits", action="store_true", default=False,
        help="Include raw interleaved/deinterleaved bit strings in output",
    )
    parser.add_argument(
        "--backend",
        choices=["native", "dsd-fme"],
        default="native",
        help="Decoder backend to use",
    )
    parser.add_argument(
        "--dsd-fme-cmd",
        default=None,
        help="Full dsd-fme command line to run when --backend=dsd-fme",
    )

    args = parser.parse_args()

    if args.backend == "dsd-fme":
        _run_dsdfme_backend(args)
    elif args.source is None and not args.gui:
        # Default: launch GUI
        from sdrparser.gui.app import launch
        launch()
    else:
        _run_headless(args)


if __name__ == "__main__":
    main()
