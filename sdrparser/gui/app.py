"""
SDRParser Tkinter GUI.

Layout
------
┌─ Toolbar ──────────────────────────────────────────────────────────┐
│  [Source ▼] [Host/File] [Port] [SR ▼] [Protocols ☑DMR ☑P25 ☑NXDN] │
│  [▶ Connect]  Status: ●  Frames: 0                                  │
├─ Notebook ─────────────────────────────────────────────────────────┤
│  Tab 1: Headers & Protocol Info                                      │
│  Tab 2: MBE Frames  (Interleaved | Deinterleaved side-by-side)      │
│  Tab 3: Raw Bit Log                                                  │
└────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, font, scrolledtext, ttk
from typing import Optional

from sdrparser.audio.input import (
    TCPClientSource,
    UDPSource,
    FileSource,
    DEFAULT_PORT,
    DEFAULT_SAMPLE_RATE,
)
from sdrparser.main import SDRParserPipeline
from sdrparser.protocols.base import DecodedFrame, FrameKind

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

BG_DARK   = "#1e1e2e"
BG_MID    = "#2a2a3e"
BG_PANEL  = "#313150"
FG_MAIN   = "#cdd6f4"
FG_ACCENT = "#89b4fa"
FG_GOOD   = "#a6e3a1"
FG_WARN   = "#fab387"
FG_VOICE  = "#89dceb"
FG_DATA   = "#f9e2af"
FG_CTRL   = "#cba6f7"
FG_INTER  = "#f38ba8"   # interleaved bits
FG_DEINT  = "#a6e3a1"   # deinterleaved bits

MONO_FONT = ("Courier New", 9)
LABEL_FONT = ("Helvetica", 9, "bold")

KIND_COLOURS = {
    "VOICE":   FG_VOICE,
    "DATA":    FG_DATA,
    "HEADER":  FG_ACCENT,
    "CONTROL": FG_CTRL,
    "UNKNOWN": FG_MAIN,
}

PROTO_COLOURS = {
    "DMR":  "#89dceb",
    "P25":  "#f9e2af",
    "NXDN": "#cba6f7",
}

MAX_LOG_LINES = 500


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class SDRParserApp(tk.Tk):
    """Root window of the SDRParser GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.title("SDRParser — DMR / P25 / NXDN Digital Voice Decoder")
        self.geometry("1100x720")
        self.configure(bg=BG_DARK)
        self.minsize(900, 580)

        self._pipeline: Optional[SDRParserPipeline] = None
        self._frame_queue: queue.Queue[DecodedFrame] = queue.Queue(maxsize=256)
        self._frame_count = 0
        self._mbe_index   = 0

        # Tkinter variables
        self._var_source   = tk.StringVar(value="TCP Client")
        self._var_host     = tk.StringVar(value="127.0.0.1")
        self._var_port     = tk.IntVar(value=DEFAULT_PORT)
        self._var_sr       = tk.StringVar(value="48000")
        self._var_baud     = tk.StringVar(value="4800")
        self._var_dmr      = tk.BooleanVar(value=True)
        self._var_p25      = tk.BooleanVar(value=True)
        self._var_nxdn     = tk.BooleanVar(value=True)
        self._var_status   = tk.StringVar(value="Idle")
        self._var_frames   = tk.StringVar(value="Frames: 0")

        self._build_toolbar()
        self._build_notebook()
        self._build_statusbar()

        self.after(100, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bg=BG_MID, pady=6, padx=8)
        bar.pack(fill="x", side="top")

        # Row 1 — source selection
        r1 = tk.Frame(bar, bg=BG_MID)
        r1.pack(fill="x", pady=(0, 4))

        tk.Label(r1, text="Source:", bg=BG_MID, fg=FG_ACCENT,
                 font=LABEL_FONT).pack(side="left", padx=(0, 4))
        src_cb = ttk.Combobox(
            r1, textvariable=self._var_source, width=12,
            values=["TCP Client", "UDP Listener", "File"],
            state="readonly",
        )
        src_cb.pack(side="left", padx=(0, 8))
        src_cb.bind("<<ComboboxSelected>>", self._on_source_changed)

        tk.Label(r1, text="Host / File:", bg=BG_MID, fg=FG_ACCENT,
                 font=LABEL_FONT).pack(side="left", padx=(0, 4))
        self._host_entry = tk.Entry(r1, textvariable=self._var_host,
                                    width=18, bg=BG_PANEL, fg=FG_MAIN,
                                    insertbackground=FG_MAIN)
        self._host_entry.pack(side="left", padx=(0, 2))

        self._browse_btn = tk.Button(
            r1, text="Browse…", command=self._browse_file,
            bg=BG_PANEL, fg=FG_ACCENT, relief="flat",
        )

        tk.Label(r1, text="Port:", bg=BG_MID, fg=FG_ACCENT,
                 font=LABEL_FONT).pack(side="left", padx=(8, 4))
        tk.Entry(r1, textvariable=self._var_port, width=7,
                 bg=BG_PANEL, fg=FG_MAIN,
                 insertbackground=FG_MAIN).pack(side="left", padx=(0, 8))

        tk.Label(r1, text="Sample Rate:", bg=BG_MID, fg=FG_ACCENT,
                 font=LABEL_FONT).pack(side="left", padx=(0, 4))
        ttk.Combobox(
            r1, textvariable=self._var_sr, width=9,
            values=["12000", "24000", "48000", "96000"],
            state="readonly",
        ).pack(side="left", padx=(0, 8))

        tk.Label(r1, text="Baud:", bg=BG_MID, fg=FG_ACCENT,
                 font=LABEL_FONT).pack(side="left", padx=(0, 4))
        ttk.Combobox(
            r1, textvariable=self._var_baud, width=7,
            values=["2400", "4800"],
            state="readonly",
        ).pack(side="left", padx=(0, 8))

        # Row 2 — protocol checkboxes + connect
        r2 = tk.Frame(bar, bg=BG_MID)
        r2.pack(fill="x")

        tk.Label(r2, text="Protocols:", bg=BG_MID, fg=FG_ACCENT,
                 font=LABEL_FONT).pack(side="left", padx=(0, 4))
        for label, var, colour in [
            ("DMR",  self._var_dmr,  PROTO_COLOURS["DMR"]),
            ("P25",  self._var_p25,  PROTO_COLOURS["P25"]),
            ("NXDN", self._var_nxdn, PROTO_COLOURS["NXDN"]),
        ]:
            tk.Checkbutton(
                r2, text=label, variable=var,
                bg=BG_MID, fg=colour, selectcolor=BG_PANEL,
                activebackground=BG_MID, activeforeground=colour,
                font=LABEL_FONT,
            ).pack(side="left", padx=4)

        self._connect_btn = tk.Button(
            r2, text="▶  Connect", command=self._toggle_connection,
            bg=FG_GOOD, fg=BG_DARK, relief="flat",
            font=("Helvetica", 9, "bold"), padx=10,
        )
        self._connect_btn.pack(side="left", padx=(16, 4))

        tk.Button(
            r2, text="🗑  Clear", command=self._clear_all,
            bg=BG_PANEL, fg=FG_WARN, relief="flat",
            font=("Helvetica", 9), padx=6,
        ).pack(side="left", padx=4)

        tk.Label(r2, textvariable=self._var_status, bg=BG_MID,
                 fg=FG_GOOD, font=LABEL_FONT).pack(side="left", padx=(16, 4))
        tk.Label(r2, textvariable=self._var_frames, bg=BG_MID,
                 fg=FG_MAIN, font=LABEL_FONT).pack(side="left", padx=(8, 0))

    def _build_notebook(self) -> None:
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("TNotebook",       background=BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab",   background=BG_MID,  foreground=FG_MAIN,
                        padding=[10, 4])
        style.map("TNotebook.Tab",
                  background=[("selected", BG_PANEL)],
                  foreground=[("selected", FG_ACCENT)])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        # Tab 1 — Headers
        self._tab_headers = tk.Frame(nb, bg=BG_DARK)
        nb.add(self._tab_headers, text=" Headers & Protocol Info ")
        self._build_header_tab()

        # Tab 2 — MBE frames
        self._tab_mbe = tk.Frame(nb, bg=BG_DARK)
        nb.add(self._tab_mbe, text=" MBE Frames (Interleaved / Deinterleaved) ")
        self._build_mbe_tab()

        # Tab 3 — Raw bit log
        self._tab_log = tk.Frame(nb, bg=BG_DARK)
        nb.add(self._tab_log, text=" Raw Frame Log ")
        self._build_log_tab()

    def _build_header_tab(self) -> None:
        f = self._tab_headers

        # Left: table of last 200 decoded headers
        left = tk.Frame(f, bg=BG_DARK)
        left.pack(side="left", fill="both", expand=True, padx=(4, 2), pady=4)

        tk.Label(left, text="Decoded Frames", bg=BG_DARK, fg=FG_ACCENT,
                 font=LABEL_FONT).pack(anchor="w")

        cols = ("Protocol", "Kind", "Fields")
        self._header_tree = ttk.Treeview(
            left, columns=cols, show="headings", height=28,
            selectmode="browse",
        )
        self._header_tree.heading("Protocol", text="Protocol")
        self._header_tree.heading("Kind",     text="Kind")
        self._header_tree.heading("Fields",   text="Header Fields")
        self._header_tree.column("Protocol", width=80,  anchor="center")
        self._header_tree.column("Kind",     width=80,  anchor="center")
        self._header_tree.column("Fields",   width=600, anchor="w")

        # Tag colours
        for proto, colour in PROTO_COLOURS.items():
            self._header_tree.tag_configure(proto, foreground=colour)

        vsb = ttk.Scrollbar(left, orient="vertical",
                            command=self._header_tree.yview)
        self._header_tree.configure(yscrollcommand=vsb.set)
        self._header_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")

        style = ttk.Style()
        style.configure("Treeview",
                        background=BG_PANEL, foreground=FG_MAIN,
                        rowheight=20, fieldbackground=BG_PANEL)
        style.configure("Treeview.Heading",
                        background=BG_MID, foreground=FG_ACCENT)
        style.map("Treeview", background=[("selected", BG_MID)])

        # Right: protocol stats
        right = tk.Frame(f, bg=BG_MID, width=200)
        right.pack(side="right", fill="y", padx=(2, 4), pady=4)
        right.pack_propagate(False)

        tk.Label(right, text="Statistics", bg=BG_MID, fg=FG_ACCENT,
                 font=LABEL_FONT).pack(anchor="w", padx=8, pady=(8, 4))

        self._stat_labels: dict[str, tk.StringVar] = {}
        for proto in ("DMR", "P25", "NXDN"):
            var = tk.StringVar(value=f"{proto}: 0 frames")
            self._stat_labels[proto] = var
            tk.Label(right, textvariable=var, bg=BG_MID,
                     fg=PROTO_COLOURS[proto],
                     font=MONO_FONT).pack(anchor="w", padx=8, pady=2)

        self._proto_counts = {"DMR": 0, "P25": 0, "NXDN": 0}

    def _build_mbe_tab(self) -> None:
        f = self._tab_mbe

        top = tk.Frame(f, bg=BG_DARK)
        top.pack(fill="x", padx=4, pady=(4, 0))

        tk.Label(top, text="Frame #:", bg=BG_DARK, fg=FG_ACCENT,
                 font=LABEL_FONT).pack(side="left")
        self._var_mbe_sel = tk.StringVar(value="Latest")
        self._mbe_index_cb = ttk.Combobox(
            top, textvariable=self._var_mbe_sel, width=10,
            values=["Latest"], state="readonly",
        )
        self._mbe_index_cb.pack(side="left", padx=4)
        self._mbe_index_cb.bind("<<ComboboxSelected>>", self._on_mbe_select)

        self._var_mbe_info = tk.StringVar(value="")
        tk.Label(top, textvariable=self._var_mbe_info, bg=BG_DARK,
                 fg=FG_MAIN, font=MONO_FONT).pack(side="left", padx=12)

        panels = tk.Frame(f, bg=BG_DARK)
        panels.pack(fill="both", expand=True, padx=4, pady=4)

        # Interleaved pane
        lf_i = tk.LabelFrame(panels, text=" Interleaved (as transmitted) ",
                              bg=BG_DARK, fg=FG_INTER, font=LABEL_FONT)
        lf_i.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self._mbe_interleaved = scrolledtext.ScrolledText(
            lf_i, bg=BG_PANEL, fg=FG_INTER, font=MONO_FONT,
            wrap="word", state="disabled", width=44,
        )
        self._mbe_interleaved.pack(fill="both", expand=True, padx=4, pady=4)

        # Deinterleaved pane
        lf_d = tk.LabelFrame(panels, text=" Deinterleaved (codec-ready) ",
                              bg=BG_DARK, fg=FG_DEINT, font=LABEL_FONT)
        lf_d.pack(side="right", fill="both", expand=True, padx=(4, 0))
        self._mbe_deinterleaved = scrolledtext.ScrolledText(
            lf_d, bg=BG_PANEL, fg=FG_DEINT, font=MONO_FONT,
            wrap="word", state="disabled", width=44,
        )
        self._mbe_deinterleaved.pack(fill="both", expand=True, padx=4, pady=4)

        self._mbe_frames_store: list[tuple[DecodedFrame, int]] = []

    def _build_log_tab(self) -> None:
        f = self._tab_log
        tk.Label(f, text="Raw Frame Log", bg=BG_DARK, fg=FG_ACCENT,
                 font=LABEL_FONT).pack(anchor="w", padx=4, pady=(4, 0))
        self._raw_log = scrolledtext.ScrolledText(
            f, bg=BG_PANEL, fg=FG_MAIN, font=MONO_FONT,
            wrap="none", state="disabled",
        )
        self._raw_log.pack(fill="both", expand=True, padx=4, pady=4)
        self._raw_log.tag_config("dmr",  foreground=PROTO_COLOURS["DMR"])
        self._raw_log.tag_config("p25",  foreground=PROTO_COLOURS["P25"])
        self._raw_log.tag_config("nxdn", foreground=PROTO_COLOURS["NXDN"])

    def _build_statusbar(self) -> None:
        sb = tk.Frame(self, bg=BG_MID, height=22)
        sb.pack(fill="x", side="bottom")
        tk.Label(sb, text="SDRParser v0.1.0 — SDR++ audio decoder for DMR, P25, NXDN",
                 bg=BG_MID, fg=FG_MAIN, font=("Helvetica", 8),
                 anchor="w").pack(side="left", padx=8)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _on_source_changed(self, _event=None) -> None:
        src = self._var_source.get()
        if src == "File":
            self._browse_btn.pack(side="left", padx=(2, 8))
        else:
            self._browse_btn.pack_forget()

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select raw PCM / WAV file",
            filetypes=[("WAV files", "*.wav"),
                       ("Raw PCM", "*.raw *.pcm"),
                       ("All files", "*.*")],
        )
        if path:
            self._var_host.set(path)

    def _toggle_connection(self) -> None:
        if self._pipeline and self._pipeline.is_running:
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def _start_pipeline(self) -> None:
        protos = []
        if self._var_dmr.get():  protos.append("DMR")
        if self._var_p25.get():  protos.append("P25")
        if self._var_nxdn.get(): protos.append("NXDN")
        if not protos:
            self._var_status.set("⚠ No protocols selected")
            return

        src_name  = self._var_source.get()
        host      = self._var_host.get()
        port      = self._var_port.get()
        sr        = int(self._var_sr.get())
        baud      = int(self._var_baud.get())

        try:
            if src_name == "TCP Client":
                source = TCPClientSource(host=host, port=port, sample_rate=sr)
            elif src_name == "UDP Listener":
                source = UDPSource(bind_host="0.0.0.0", port=port, sample_rate=sr)
            else:
                source = FileSource(path=host, sample_rate=sr)
        except Exception as exc:
            self._var_status.set(f"⚠ {exc}")
            return

        self._pipeline = SDRParserPipeline(
            source=source,
            on_frame=self._on_frame,
            baud_rate=baud,
            enabled_protocols=protos,
        )
        self._pipeline.start()

        self._connect_btn.config(text="■  Disconnect", bg=FG_WARN)
        self._var_status.set(f"● Connected ({src_name})")

    def _stop_pipeline(self) -> None:
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        self._connect_btn.config(text="▶  Connect", bg=FG_GOOD)
        self._var_status.set("Idle")

    def _clear_all(self) -> None:
        # Headers tab
        for item in self._header_tree.get_children():
            self._header_tree.delete(item)
        self._proto_counts = {"DMR": 0, "P25": 0, "NXDN": 0}
        for proto, var in self._stat_labels.items():
            var.set(f"{proto}: 0 frames")
        # MBE tab
        self._mbe_frames_store.clear()
        self._mbe_index_cb.configure(values=["Latest"])
        self._var_mbe_sel.set("Latest")
        self._clear_mbe_panes()
        # Log tab
        self._raw_log.configure(state="normal")
        self._raw_log.delete("1.0", "end")
        self._raw_log.configure(state="disabled")
        # Counters
        self._frame_count = 0
        self._mbe_index   = 0
        self._var_frames.set("Frames: 0")

    # ------------------------------------------------------------------
    # Frame callback (called from worker thread → queued → GUI thread)
    # ------------------------------------------------------------------

    def _on_frame(self, frame: DecodedFrame) -> None:
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def _poll_queue(self) -> None:
        """Process up to 20 queued frames per tick."""
        for _ in range(20):
            try:
                frame = self._frame_queue.get_nowait()
            except queue.Empty:
                break
            self._display_frame(frame)
        self.after(50, self._poll_queue)

    # ------------------------------------------------------------------
    # Frame display
    # ------------------------------------------------------------------

    def _display_frame(self, frame: DecodedFrame) -> None:
        self._frame_count += 1
        self._var_frames.set(f"Frames: {self._frame_count}")

        proto = frame.protocol
        if proto in self._proto_counts:
            self._proto_counts[proto] += 1
            self._stat_labels[proto].set(
                f"{proto}: {self._proto_counts[proto]} frames"
            )

        # Header table
        fields_str = "  |  ".join(f"{k}: {v}" for k, v in frame.header_fields)
        tag = proto.lower()
        self._header_tree.insert(
            "", "end",
            values=(proto, frame.kind.name, fields_str),
            tags=(proto,),
        )
        # Keep tree trim
        children = self._header_tree.get_children()
        while len(children) > 300:
            self._header_tree.delete(children[0])
            children = self._header_tree.get_children()
        # Auto-scroll
        if children:
            self._header_tree.see(children[-1])

        # MBE frames
        for mf in frame.mbe_frames:
            entry_label = (
                f"{self._mbe_index + 1}  "
                f"[{proto} {mf.frame_type.name}]"
            )
            self._mbe_frames_store.append((frame, len(self._mbe_frames_store)))
            # Keep at most 200 stored frames
            if len(self._mbe_frames_store) > 200:
                self._mbe_frames_store.pop(0)
            self._mbe_index += 1

            vals = ["Latest"] + [
                f"{i + 1}  [{s.protocol} {mf.frame_type.name}]"
                for i, (s, _) in enumerate(self._mbe_frames_store)
            ]
            self._mbe_index_cb.configure(values=vals)

            # Auto-display if "Latest" is selected
            if self._var_mbe_sel.get() == "Latest":
                self._show_mbe(frame, mf)

        # Raw log
        summary = frame.summary()
        self._raw_log.configure(state="normal")
        self._raw_log.insert("end", summary + "\n", tag)
        # Trim log
        line_count = int(self._raw_log.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            self._raw_log.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")
        self._raw_log.see("end")
        self._raw_log.configure(state="disabled")

    def _show_mbe(self, frame: DecodedFrame, mf) -> None:
        """Populate both MBE panes for one MBEFrame."""
        info = (
            f"Protocol: {mf.protocol}  |  Codec: {mf.frame_type.name}  |  "
            f"Frame #{mf.frame_index}  |  "
            f"Bits: {len(mf.interleaved_bits)}"
        )
        self._var_mbe_info.set(info)

        inter_text = self._format_bits(mf.interleaved_bits, label="Interleaved")
        deint_text = self._format_bits(mf.deinterleaved_bits, label="Deinterleaved")

        self._update_text(self._mbe_interleaved, inter_text)
        self._update_text(self._mbe_deinterleaved, deint_text)

    @staticmethod
    def _format_bits(bits: list, label: str) -> str:
        """Format a bit list as groups of 8, with hex and position markers."""
        lines = [f"── {label} ({len(bits)} bits) ──\n"]
        for i in range(0, len(bits), 8):
            chunk = bits[i:i + 8]
            bin_str = " ".join(str(b) for b in chunk)
            byte_val = 0
            for b in chunk:
                byte_val = (byte_val << 1) | b
            hex_str = f"0x{byte_val:02X}" if len(chunk) == 8 else "    "
            lines.append(f"  [{i:3d}]  {bin_str:<23}  {hex_str}\n")
        return "".join(lines)

    @staticmethod
    def _update_text(widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    def _clear_mbe_panes(self) -> None:
        self._var_mbe_info.set("")
        self._update_text(self._mbe_interleaved, "")
        self._update_text(self._mbe_deinterleaved, "")

    def _on_mbe_select(self, _event=None) -> None:
        sel = self._var_mbe_sel.get()
        if sel == "Latest" or not self._mbe_frames_store:
            return
        try:
            idx = int(sel.split()[0]) - 1
            if 0 <= idx < len(self._mbe_frames_store):
                frame, _ = self._mbe_frames_store[idx]
                if frame.mbe_frames:
                    self._show_mbe(frame, frame.mbe_frames[0])
        except (ValueError, IndexError):
            pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        self._stop_pipeline()
        self.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch() -> None:
    """Launch the SDRParser GUI application."""
    app = SDRParserApp()
    app.mainloop()


if __name__ == "__main__":
    launch()
