# SDRParser

A Python application that processes raw audio from **SDR++** and parses the
data stream for the **DMR**, **NXDN**, and **P25** digital voice formats.

It outputs formatted header data, raw header HEX, and **MBE codec frames** in both
**interleaved** (as transmitted) and **deinterleaved** (codec-ready) form
through a dark-themed Tkinter GUI or a headless CLI.

---

## Features

| Feature | Detail |
|---|---|
| **SDR++ integration** | Connects to SDR++'s *Network Sink* module via **TCP** (client) or **UDP** (listener) |
| **DMR decoding** | Burst sync, BS/MS voice and data frames, LC header parsing (FLCO, Dst/Src IDs), two AMBE+2 frames per burst |
| **P25 Phase 1 decoding** | Frame sync, NID (NAC + DUID), HDU / LDU1 / LDU2 header parsing, nine IMBE frames per LDU |
| **NXDN decoding** | Frame sync, LICH (RFCT / FT), RDCH header parsing (Dst/Src IDs, Msg Type), AMBE+2 frame extraction |
| **MBE interleaving** | All frames displayed in both interleaved (OTA) and deinterleaved (codec-input) bit order |
| **HEX output** | Raw protocol header bits and MBE frames are emitted as uppercase byte HEX strings (dsd-fme style) |
| **Tkinter GUI** | Multi-tab dark UI: Headers table · MBE side-by-side viewer · Raw frame log |
| **CLI / headless** | Stream-to-stdout mode for scripting and logging |
| **File input** | Analyse saved WAV or raw PCM recordings |

---

## SDR++ Setup

1. Open SDR++ and tune to a DMR / P25 / NXDN transmission using **NFM** demodulation.
2. Go to **Menu → Network Sink** and configure:

   | Setting | Value |
   |---|---|
   | Protocol | **TCP** (SDRParser connects to SDR++) or **UDP** (SDR++ sends to SDRParser) |
   | Hostname | `127.0.0.1` (or your machine's IP) |
   | Port | `7355` (default) |
   | Sample Rate | `48000` Hz |
   | Stereo | Off (mono recommended) |

3. Click **Start** in the Network Sink panel.
4. Launch SDRParser and press **Connect**.

### Wire format

SDR++ network_sink sends raw **signed 16-bit PCM** (`int16_t`, little-endian),
scaled at ×32 768 relative to a ±1.0 float range.  There are no packet
headers — it is a raw PCM byte stream over TCP or UDP.  UDP packets contain
512 samples (1 024 bytes) per datagram.

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: `numpy`, `scipy` (stdlib `tkinter` for the GUI).

---

## Usage

### Launch the GUI (default)

```bash
python run.py
```

### Connect to SDR++ TCP network_sink

```bash
python run.py --source tcp --host 127.0.0.1 --port 7355
```

### Parse a direct TCP dibit/bitstream feed

Use this when your upstream tool sends already-demodulated dibits/bits and
you want SDRParser to only perform protocol framing and header/voice payload
reassembly:

```bash
python run.py --source tcp-bits --host 127.0.0.1 --port 7355 --bit-format auto
```

`--bit-format` accepts `auto`, `ascii-bits`, `dibit-bytes`, or `packed-dibits`.

### Listen for SDR++ UDP datagrams

```bash
python run.py --source udp --port 7355
```

### Decode a saved WAV file

```bash
python run.py --source file --file recording.wav
```

### Restrict to one protocol

```bash
python run.py --source tcp --protocols DMR
```

### Use dsd-fme as decoder backend (recommended for exact parity)

This mode runs dsd-fme directly and filters its output to only:

* transmission header lines
* AMBE/IMBE vocoder hex frames

```bash
python run.py \
  --backend dsd-fme \
  --dsd-fme-cmd "dsd-fme <your-normal-dsd-fme-args>"
```

Example using a WAV input passed to dsd-fme:

```bash
python run.py \
  --backend dsd-fme \
  --dsd-fme-cmd "dsd-fme -r recording.wav"
```

### NXDN 6.25 kHz (2 400 baud) narrow-band mode

```bash
python run.py --source tcp --protocols NXDN --baud 2400
```

---

## Project Structure

```
SDRParser/
├── run.py                        Entry point (GUI + CLI)
├── requirements.txt
├── setup.py
├── sdrparser/
│   ├── audio/
│   │   └── input.py              TCPClientSource, UDPSource, FileSource
│   ├── dsp/
│   │   └── demod.py              SymbolDemodulator (4-FSK), BitStreamBuffer
│   ├── mbe/
│   │   └── frames.py             MBEFrame, interleave/deinterleave tables
│   ├── protocols/
│   │   ├── base.py               DecodedFrame, ProtocolDecoder base class
│   │   ├── dmr.py                DMR burst decoder
│   │   ├── p25.py                P25 Phase 1 decoder
│   │   └── nxdn.py               NXDN decoder
│   ├── gui/
│   │   └── app.py                Tkinter GUI
│   └── main.py                   SDRParserPipeline (audio→DSP→decoders)
└── tests/
    ├── test_mbe.py               MBE interleave/deinterleave unit tests
    ├── test_dmr.py               DMR decoder unit tests
    ├── test_p25.py               P25 decoder unit tests
    └── test_nxdn.py              NXDN decoder unit tests
```

---

## GUI Overview

### Headers & Protocol Info tab

Displays every decoded frame with protocol colour-coding (DMR=cyan,
P25=yellow, NXDN=purple), frame kind (VOICE / HEADER / DATA / CONTROL),
and all parsed header fields.  A statistics panel on the right shows
per-protocol frame counts.

### MBE Frames (Interleaved / Deinterleaved) tab

Shows the latest (or any selected) MBE codec frame in two panels:

* **Left (red)** — bits exactly as they appear in the over-the-air burst
  (interleaved form, ready for error-correction).
* **Right (green)** — bits after applying the protocol-specific
  deinterleave permutation (natural codec order, suitable for feeding to
  an AMBE+2 / IMBE chip or software decoder).

Each row shows the bit-position offset, 8 bits, and the byte hex value.

### Raw Frame Log tab

Scrolling text log of every decoded frame summary for logging / debugging.

---

## Supported Codecs

| Protocol | Codec | Frame size | Frames per burst |
|---|---|---|---|
| DMR | AMBE+2 | 72 bits | 2 per 30 ms burst |
| P25 Phase 1 | IMBE | 88 bits | 9 per LDU |
| NXDN | AMBE+2 | 72 bits | 1 per frame |

---

## Running Tests

```bash
python -m unittest discover -v tests/
```

75 tests covering interleave tables, round-trip correctness, header
parsing, and full decoder frame detection for all three protocols.
