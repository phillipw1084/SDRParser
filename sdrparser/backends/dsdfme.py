"""dsd-fme backend integration helpers.

This backend does not decode frames itself. It runs dsd-fme and parses
its textual output into structured header/vocoder events.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
from typing import Generator, Iterable, Optional


_PROTOCOL_RE = re.compile(r"\b(DMR|P25|NXDN|D-STAR|YSF)\b", re.IGNORECASE)
_AMBE_RE = re.compile(r"\bAMBE\b[^0-9A-F]*([0-9A-F]{12,16})\b", re.IGNORECASE)
_IMBE_RE = re.compile(r"\bIMBE\b[^0-9A-F]*([0-9A-F]{20,24})\b", re.IGNORECASE)

_HEADER_HINTS = (
    "voice",
    "call",
    "group",
    "private",
    "source",
    "src",
    "target",
    "dst",
    "talkgroup",
    "tg",
    "nac",
    "ran",
    "color code",
    "slot",
    "flco",
    "fid",
    "pi",
    "encrypted",
    "clear",
    "alg",
)


@dataclass
class DsdfmeEvent:
    kind: str
    protocol: str
    text: str
    vocoder_type: str = ""
    vocoder_hex: str = ""


class DsdfmeCommandError(RuntimeError):
    """Raised when dsd-fme fails to start."""


def parse_dsd_fme_line(line: str, last_protocol: str = "Unknown") -> tuple[list[DsdfmeEvent], str]:
    """Parse one dsd-fme output line into header/vocoder events."""
    cleaned = line.strip()
    if not cleaned:
        return [], last_protocol

    proto_match = _PROTOCOL_RE.search(cleaned)
    protocol = proto_match.group(1).upper() if proto_match else last_protocol

    events: list[DsdfmeEvent] = []

    ambe_match = _AMBE_RE.search(cleaned)
    if ambe_match:
        events.append(
            DsdfmeEvent(
                kind="vocoder",
                protocol=protocol,
                text=cleaned,
                vocoder_type="AMBE",
                vocoder_hex=ambe_match.group(1).upper(),
            )
        )

    imbe_match = _IMBE_RE.search(cleaned)
    if imbe_match:
        events.append(
            DsdfmeEvent(
                kind="vocoder",
                protocol=protocol,
                text=cleaned,
                vocoder_type="IMBE",
                vocoder_hex=imbe_match.group(1).upper(),
            )
        )

    lower = cleaned.lower()
    if not events and any(hint in lower for hint in _HEADER_HINTS):
        events.append(
            DsdfmeEvent(
                kind="header",
                protocol=protocol,
                text=cleaned,
            )
        )

    return events, protocol


def stream_dsdfme_events(command: Iterable[str]) -> Generator[DsdfmeEvent, None, int]:
    """Run dsd-fme command and stream parsed events.

    Returns the subprocess return code at generator completion.
    """
    cmd = list(command)
    if not cmd:
        raise DsdfmeCommandError("Empty dsd-fme command")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise DsdfmeCommandError(str(exc)) from exc

    assert proc.stdout is not None
    last_protocol = "Unknown"
    for line in proc.stdout:
        events, last_protocol = parse_dsd_fme_line(line, last_protocol)
        for event in events:
            yield event

    return_code = proc.wait()
    return return_code
