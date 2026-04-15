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
import signal
import sys
import time

from sdrparser.audio.input import (
    DEFAULT_PORT,
    DEFAULT_SAMPLE_RATE,
    FileSource,
    TCPClientSource,
    UDPSource,
)
from sdrparser.main import SDRParserPipeline
from sdrparser.protocols.base import DecodedFrame


def _print_frame(frame: DecodedFrame) -> None:
    fields = "  |  ".join(f"{k}: {v}" for k, v in frame.header_fields)
    print(f"[{frame.protocol:4s}] {frame.kind.name:<8s} {fields}")
    header_hex = frame.header_hex()
    if header_hex:
        print(f"         Header HEX   : {header_hex}")
    for mf in frame.mbe_frames:
        inter = " ".join(str(b) for b in mf.interleaved_bits)
        deint = " ".join(str(b) for b in mf.deinterleaved_bits)
        inter_hex = mf.bits_hex("interleaved")
        deint_hex = mf.bits_hex("deinterleaved")
        print(f"         MBE#{mf.frame_index} [{mf.frame_type.name}]")
        print(f"           Interleaved HEX  : {inter_hex}")
        print(f"           Deinterleaved HEX: {deint_hex}")
        print(f"           Interleaved  : {inter}")
        print(f"           Deinterleaved: {deint}")


def _run_headless(args: argparse.Namespace) -> None:
    protos = [p.strip().upper() for p in args.protocols.split(",")]

    if args.source == "tcp":
        source = TCPClientSource(
            host=args.host, port=args.port, sample_rate=args.sample_rate
        )
    elif args.source == "udp":
        source = UDPSource(
            bind_host="0.0.0.0", port=args.port, sample_rate=args.sample_rate
        )
    elif args.source == "file":
        if not args.file:
            print("Error: --file is required when --source=file", file=sys.stderr)
            sys.exit(1)
        source = FileSource(path=args.file, sample_rate=args.sample_rate)
    else:
        print(f"Unknown source: {args.source}", file=sys.stderr)
        sys.exit(1)

    pipeline = SDRParserPipeline(
        source=source,
        on_frame=_print_frame,
        baud_rate=args.baud,
        enabled_protocols=protos,
    )

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SDRParser — DMR/P25/NXDN digital voice decoder for SDR++ audio"
    )
    parser.add_argument(
        "--gui", action="store_true", default=False,
        help="Launch the GUI (default when no --source is given)"
    )
    parser.add_argument(
        "--source", choices=["tcp", "udp", "file"], default=None,
        help="Audio input source (omit to launch GUI)"
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

    args = parser.parse_args()

    if args.source is None and not args.gui:
        # Default: launch GUI
        from sdrparser.gui.app import launch
        launch()
    else:
        _run_headless(args)


if __name__ == "__main__":
    main()
