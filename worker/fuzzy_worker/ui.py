"""The window. A guided flow, in the look of the Fuzzy site: Link -> Get ready -> Share.

Nothing slow runs on the window's own thread (talking to the server, finding the graphics card,
installing, stopping): those run in the background and report back through a queue, so the
window never freezes.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk
from typing import Any

from fuzzy_worker.core import (
    IMAGE_BACKENDS,
    VERSION,
    Config,
    JobError,
    Worker,
    check_link,
    gpu_info,
    load_config,
    save_config,
)
from fuzzy_worker.setup import (
    SetupError,
    describe,
    find_ollama,
    parse_code,
    recommend,
    run_setup,
    set_autostart,
)

# The Fuzzy site's palette (its light theme).
BG, PANEL, INK, MUTED, LINE = "#faf8f4", "#ffffff", "#1f1f1c", "#6d6b64", "#d9d4c8"
ACCENT, ACCENT_INK, SOFT, WARN, AMBER = "#2f6b57", "#ffffff", "#f1ede4", "#9a3b2e", "#b7791f"
FONT = "Segoe UI"

MODELS = ("qwen3:4b", "qwen3:8b", "qwen3:14b")
TRUST_WORDS = {
    "private": "your own computer: it may run Fuzzy's thinking",
    "shared": "a shared computer: it only draws Fuzzy's public pictures",
}


def _whole(text: str) -> int:
    try:
        return max(0, int(text.strip() or 0))
    except ValueError:
        return 0


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(f"Fuzzy Worker {VERSION}")
        root.geometry("660x780")
        root.minsize(600, 700)
        root.configure(bg=BG)
        self.config = load_config()
        self.worker: Worker | None = None
        self.events: queue.Queue[tuple[str, Any, Any]] = queue.Queue()
        self.busy = False  # linking or setting up
        self.gpu: dict[str, Any] = {}
        self._style()
        self._build()
        self._refresh()
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(250, self._pump)
        self._background(gpu_info, "gpu")  # nvidia-smi can take seconds: never on this thread
        if self.config.connect_on_start and self._ready():
            root.after(800, self._toggle)

    # ------------------------------------------------------------------ style
    def _style(self) -> None:
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=INK, font=(FONT, 10), fieldbackground=PANEL)
        s.configure("TFrame", background=BG)
        s.configure(
            "Card.TFrame", background=PANEL, relief="solid", borderwidth=1, bordercolor=LINE
        )
        s.configure("Inner.TFrame", background=PANEL, borderwidth=0, relief="flat")
        s.configure("TLabel", background=BG, foreground=INK)
        s.configure("Card.TLabel", background=PANEL)
        s.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        s.configure("Title.TLabel", background=PANEL, font=(FONT, 12, "bold"))
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure(
            "TNotebook.Tab", padding=(16, 7), background=SOFT, foreground=MUTED, borderwidth=0
        )
        s.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", INK)])
        s.configure("TButton", padding=(14, 7), background=PANEL, bordercolor=LINE, relief="solid")
        s.map("TButton", background=[("active", SOFT)])
        s.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground=ACCENT_INK,
            bordercolor=ACCENT,
            font=(FONT, 10, "bold"),
        )
        s.map(
            "Accent.TButton",
            background=[("active", "#285a49"), ("disabled", LINE)],
            foreground=[("disabled", MUTED)],
        )
        s.configure("TCheckbutton", background=PANEL)
        s.configure(
            "Horizontal.TProgressbar", background=ACCENT, troughcolor=SOFT, bordercolor=LINE
        )
        s.configure("TEntry", padding=6, bordercolor=LINE)
        s.configure("TCombobox", padding=5)

    # ----------------------------------------------------------------- layout
    def _card(self, parent: tk.Misc, number: str, title: str) -> tuple[ttk.Frame, ttk.Label]:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=14)
        frame.pack(fill="x", pady=(0, 12))
        head = ttk.Frame(frame, style="Inner.TFrame")
        head.pack(fill="x")
        badge = ttk.Label(head, text=number, style="Title.TLabel", foreground=MUTED, width=2)
        badge.pack(side="left")
        ttk.Label(head, text=title, style="Title.TLabel").pack(side="left")
        return frame, badge

    def _build(self) -> None:
        top = tk.Frame(self.root, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        top.pack(fill="x")
        tk.Label(top, text="Fuzzy Worker", bg=PANEL, fg=INK, font=(FONT, 15, "bold")).pack(
            side="left", padx=16, pady=(12, 2)
        )
        tk.Label(
            top, text="Lend this computer to Project Fuzzy", bg=PANEL, fg=MUTED, font=(FONT, 9)
        ).pack(side="left", pady=(18, 0))
        self.chip = tk.Label(top, text="●  Stopped", bg=PANEL, fg=MUTED, font=(FONT, 10, "bold"))
        self.chip.pack(side="right", padx=16)

        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True, padx=14, pady=12)
        home, settings, activity = (ttk.Frame(tabs, padding=(0, 12, 0, 0)) for _ in range(3))
        tabs.add(home, text="Home")
        tabs.add(settings, text="Settings")
        tabs.add(activity, text="Activity")
        self._home(home)
        self._settings(settings)
        self._activity(activity)

    def _home(self, f: ttk.Frame) -> None:
        # 1. Link
        card, self.badge1 = self._card(f, "1", "Link to your Fuzzy account")
        self.link_note = ttk.Label(card, style="Muted.TLabel", wraplength=560, justify="left")
        self.link_note.pack(anchor="w", pady=(6, 6))
        self.link_row = ttk.Frame(card, style="Inner.TFrame")
        self.link_row.pack(fill="x")
        self.code = tk.StringVar()
        ttk.Entry(self.link_row, textvariable=self.code).pack(side="left", fill="x", expand=True)
        self.link_button = ttk.Button(
            self.link_row, text="Link this computer", style="Accent.TButton", command=self._link
        )
        self.link_button.pack(side="left", padx=(8, 0))
        self.link_result = ttk.Label(card, style="Card.TLabel", wraplength=560, justify="left")
        self.link_result.pack(anchor="w", pady=(8, 0))

        # 2. Get ready
        card, self.badge2 = self._card(f, "2", "Get this computer ready")
        self.gpu_label = ttk.Label(card, text="Looking at your graphics card…", style="Card.TLabel")
        self.gpu_label.pack(anchor="w", pady=(6, 0))
        self.why_label = ttk.Label(card, style="Muted.TLabel", wraplength=560, justify="left")
        self.why_label.pack(anchor="w")
        row = ttk.Frame(card, style="Inner.TFrame")
        row.pack(fill="x", pady=8)
        ttk.Label(row, text="Model", style="Card.TLabel").pack(side="left")
        self.model = tk.StringVar(value="qwen3:8b")
        ttk.Combobox(row, textvariable=self.model, values=MODELS, state="readonly", width=12).pack(
            side="left", padx=8
        )
        self.use_cpu = tk.BooleanVar(value=self.config.use_cpu)
        ttk.Checkbutton(row, text="Also share my processor", variable=self.use_cpu).pack(
            side="left", padx=(12, 0)
        )
        self.setup_button = ttk.Button(
            card, text="Set everything up", style="Accent.TButton", command=self._setup
        )
        self.setup_button.pack(anchor="w")
        self.bar = ttk.Progressbar(card, maximum=100)
        self.bar.pack(fill="x", pady=(10, 2))
        self.step = ttk.Label(card, style="Muted.TLabel", wraplength=560, justify="left")
        self.step.pack(anchor="w")

        # 3. Share
        card, self.badge3 = self._card(f, "3", "Share")
        self.run_note = ttk.Label(card, style="Muted.TLabel", wraplength=560, justify="left")
        self.run_note.pack(anchor="w", pady=(6, 8))
        self.button = ttk.Button(
            card, text="Start sharing", style="Accent.TButton", command=self._toggle
        )
        self.button.pack(anchor="w")
        ttk.Label(
            card,
            text="Fuzzy can't run programs, read your files or reach your network. Stop (or "
            "close this window) and it disconnects right away.",
            style="Muted.TLabel",
            wraplength=560,
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

    def _settings(self, f: ttk.Frame) -> None:
        c = self.config
        card = ttk.Frame(f, style="Card.TFrame", padding=14)
        card.pack(fill="x")
        ttk.Label(card, text="Stay out of my way", style="Title.TLabel").pack(anchor="w")
        ttk.Label(card, text="0 turns a limit off.", style="Muted.TLabel").pack(
            anchor="w", pady=(0, 6)
        )
        self.idle = self._field(
            card,
            "Only after this many minutes without keyboard or mouse",
            str(c.only_when_idle_minutes),
        )
        self.gpu_limit = self._field(
            card, "Pause if my graphics card is busier than (%)", str(c.pause_above_gpu_percent)
        )
        self.cpu_percent = self._field(
            card, "Processor share, if I turn it on (%)", str(c.cpu_percent)
        )
        self.auto = tk.BooleanVar(value=c.connect_on_start)
        self.winstart = tk.BooleanVar(value=c.start_with_windows)
        ttk.Checkbutton(
            card, text="Connect automatically when this program starts", variable=self.auto
        ).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(
            card, text="Start this program when Windows starts", variable=self.winstart
        ).pack(anchor="w")

        pics = ttk.Frame(f, style="Card.TFrame", padding=14)
        pics.pack(fill="x", pady=12)
        ttk.Label(pics, text="Pictures (optional)", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            pics,
            text="To draw pictures for Fuzzy too, run Stable Diffusion WebUI with --api, then "
            "pick it here.",
            style="Muted.TLabel",
            wraplength=560,
        ).pack(anchor="w", pady=(0, 6))
        self.image_backend = tk.StringVar(value=c.image_backend)
        ttk.Combobox(
            pics, textvariable=self.image_backend, values=IMAGE_BACKENDS, state="readonly", width=16
        ).pack(anchor="w")
        self.image_url = self._field(pics, "Picture server address", c.image_url)

    def _activity(self, f: ttk.Frame) -> None:
        self.log = tk.Text(
            f,
            height=20,
            state="disabled",
            wrap="word",
            bg=PANEL,
            fg=INK,
            relief="solid",
            bd=1,
            highlightthickness=0,
            font=("Consolas", 9),
            padx=8,
            pady=6,
        )
        self.log.pack(fill="both", expand=True)

    def _field(self, parent: tk.Misc, label: str, value: str) -> tk.StringVar:
        row = ttk.Frame(parent, style="Inner.TFrame")
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, style="Card.TLabel").pack(side="left")
        var = tk.StringVar(value=value)
        ttk.Entry(row, textvariable=var, width=26).pack(side="right")
        return var

    # ------------------------------------------------------------------ state
    def _linked(self) -> bool:
        return bool(self.config.server_url and self.config.token and self.config.linked_name)

    def _prepared(self) -> bool:
        c = self.config
        return bool(c.run_llm and c.llm_model and (not c.use_cpu or c.embed_model))

    def _ready(self) -> bool:
        return self._linked() and self._prepared()

    @staticmethod
    def _tick(done: bool) -> str:
        return "✓" if done else "•"

    def _refresh(self) -> None:
        """Make every card say what is true now and what to do next."""

        c = self.config
        linked, prepared = self._linked(), self._prepared()
        self.badge1.configure(text=self._tick(linked), foreground=ACCENT if linked else MUTED)
        self.badge2.configure(text=self._tick(prepared), foreground=ACCENT if prepared else MUTED)
        ready = self._ready()
        self.badge3.configure(text="▶" if ready else "•", foreground=ACCENT if ready else MUTED)
        if linked:
            where = TRUST_WORDS.get(c.linked_trust, "")
            self.link_note.configure(text=f"Linked as “{c.linked_name}”: {where}.")
            self.link_result.configure(text="✓ Verified with Fuzzy.", foreground=ACCENT)
            self.link_button.configure(text="Use a different code")
        else:
            self.link_note.configure(
                text="On Fuzzy's Compute page, press “Add a computer” and copy the code it "
                "shows. Paste it here."
            )
            self.link_button.configure(text="Link this computer")
        _, why = recommend(self.gpu.get("gpu_memory_mb"))
        self.gpu_label.configure(
            text="Graphics card: " + (describe(self.gpu) if self.gpu else "none found")
        )
        self.why_label.configure(text=why)
        if prepared:
            extra = " and processor sharing" if c.use_cpu and c.embed_model else ""
            self.step.configure(text=f"✓ Ready: {c.llm_model}{extra}")
            self.bar["value"] = 100
        elif find_ollama() is None and not self.busy:
            self.step.configure(
                text="Ollama (the program that runs the model) isn't installed yet. Setup "
                "installs it for you."
            )
        running = self.worker is not None and self.worker.running
        self.setup_button.configure(state="disabled" if self.busy or running else "normal")
        self.link_button.configure(state="disabled" if self.busy or running else "normal")
        if running:
            return
        self.button.configure(
            state="normal" if ready and not self.busy else "disabled", text="Start sharing"
        )
        if not linked:
            self.run_note.configure(text="Link this computer first (step 1).")
        elif not prepared:
            self.run_note.configure(text="Get this computer ready first (step 2).")
        else:
            self.run_note.configure(
                text="Everything's ready. Press Start whenever you'd like Fuzzy to be able to "
                "use this computer."
            )

    def _set_chip(self, text: str) -> None:
        lowered = text.lower()
        color = MUTED
        if lowered.startswith("working"):
            color, label = ACCENT, "Working for Fuzzy"
        elif lowered.startswith("connected"):
            color, label = ACCENT, "Connected"
        elif lowered.startswith("waiting"):
            color, label = AMBER, "Paused: " + text.split(":", 1)[-1].strip()
        elif "can't reach" in lowered or lowered.startswith("the server said"):
            color, label = WARN, "No connection"
        elif lowered.startswith("stopped:"):
            color, label = WARN, text.split(":", 1)[1].strip().capitalize()
        elif lowered.startswith("starting"):
            color, label = AMBER, "Connecting…"
        else:
            label = "Stopped"
        self.chip.configure(text=f"●  {label}"[:72], fg=color)

    # ---------------------------------------------------------------- actions
    def _background(self, work: Callable[[], Any], kind: str) -> None:
        def run() -> None:
            try:
                self.events.put((kind, work(), None))
            except Exception as exc:
                self.events.put((kind + "_error", exc, None))

        threading.Thread(target=run, name=kind, daemon=True).start()

    def _link(self) -> None:
        if self._linked():  # "Use a different code"
            self.config.linked_name = ""
            self.link_result.configure(text="")
            self._refresh()
            return
        parsed = parse_code(self.code.get())
        if parsed is None:
            self.link_result.configure(
                text="That doesn't look like a connection code. Copy the whole code from "
                "Fuzzy's Compute page.",
                foreground=WARN,
            )
            return
        server, token = parsed
        self.busy = True
        self.link_result.configure(text="Checking with Fuzzy…", foreground=MUTED)
        self._refresh()
        self._background(lambda: (server, token, check_link(server, token)), "link")

    def _read(self) -> Config:
        c = self.config
        c.only_when_idle_minutes = _whole(self.idle.get())
        c.pause_above_gpu_percent = _whole(self.gpu_limit.get())
        c.cpu_percent = max(5, min(100, _whole(self.cpu_percent.get()) or 50))
        c.use_cpu = self.use_cpu.get()
        c.image_backend = self.image_backend.get()
        c.image_url = self.image_url.get().strip()
        c.connect_on_start = self.auto.get()
        c.start_with_windows = self.winstart.get()
        return c

    def _toggle(self) -> None:
        if self.worker is not None and self.worker.running:
            self.worker.stop(wait=False)  # returns at once; the chip shows when it has stopped
            self.button.configure(state="disabled", text="Stopping…")
            self.run_note.configure(
                text="Finishing up. This takes a few seconds, or until a job in progress is done."
            )
            return
        config = self._read()
        worker = Worker(
            config,
            on_status=lambda s: self.events.put(("status", s, None)),
            on_log=lambda s: self.events.put(("log", s, None)),
        )
        problem = worker.problem()
        if problem:
            messagebox.showwarning("Fuzzy Worker", problem)
            return
        save_config(config)
        set_autostart(config.start_with_windows)
        self.worker = worker
        worker.start()
        self.button.configure(text="Stop sharing")
        self.run_note.configure(text="Sharing. Fuzzy can use this computer while it's free.")
        self._refresh()

    def _setup(self) -> None:
        if self.busy:
            return
        if find_ollama() is None and not messagebox.askyesno(
            "Fuzzy Worker",
            "Ollama (the program that runs the model) isn't installed. Download it from "
            "ollama.com and install it for you now? It's a few hundred MB, and I'll keep its "
            "windows out of your way.",
        ):
            return
        self.busy = True
        config = self._read()
        tag = self.model.get()
        self.bar.configure(mode="indeterminate")
        self._refresh()

        def work() -> str:
            return run_setup(
                config,
                lambda text, fraction: self.events.put(("step", text, fraction)),
                model=tag,
            )

        self._background(work, "setup")

    # ------------------------------------------------------------------ events
    def _pump(self) -> None:
        while True:
            try:
                kind, value, extra = self.events.get_nowait()
            except queue.Empty:
                break
            self._handle(kind, value, extra)
        stopped = self.worker is not None and not self.worker.running
        if stopped and self.button["text"] in ("Stopping…", "Stop sharing"):
            self._set_chip("stopped")
            self._refresh()  # it has finished stopping (or ended by itself)
        self.root.after(250, self._pump)

    def _handle(self, kind: str, value: Any, extra: Any) -> None:
        if kind == "status":
            self._set_chip(value)
        elif kind == "log":
            self._write_log(value)
        elif kind == "gpu":
            self.gpu = value
            self.model.set(recommend(value.get("gpu_memory_mb"))[0])
            self._refresh()
        elif kind == "step":
            self.step.configure(text=value)
            if extra is None:
                self.bar.configure(mode="indeterminate")
                self.bar.start(15)
            else:
                self.bar.stop()
                self.bar.configure(mode="determinate", value=extra * 100)
            if extra in (None, 0.0, 1.0):
                self._write_log(value)
        elif kind == "link":
            server, token, data = value
            self.config.server_url, self.config.token = server, token
            self.config.linked_name = str(data.get("name", ""))
            self.config.linked_trust = str(data.get("trust", ""))
            save_config(self.config)
            self.code.set("")
            self.busy = False
            self._write_log(f"Linked as {self.config.linked_name}.")
            self._refresh()
        elif kind == "link_error":
            self.busy = False
            known = isinstance(value, JobError)
            message = str(value) if known else f"Something went wrong ({type(value).__name__})."
            self._refresh()
            self.link_result.configure(text=message, foreground=WARN)
        elif kind == "setup":
            self.bar.stop()
            self.busy = False
            self.bar.configure(mode="determinate", value=100)
            save_config(self.config)
            self._write_log(value)
            self._refresh()
            tail = "" if self._linked() else " Now link it in step 1."
            messagebox.showinfo("Fuzzy Worker", "This computer is ready." + tail)
        elif kind == "setup_error":
            self.bar.stop()
            self.busy = False
            self.bar.configure(mode="determinate", value=0)
            if isinstance(value, SetupError):
                text = str(value)
            else:
                text = f"Something went wrong ({type(value).__name__}: {value})"
            self._refresh()
            self.step.configure(text=text)
            self._write_log(text)
            messagebox.showwarning("Fuzzy Worker", text)

    def _write_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _close(self) -> None:
        if self.worker is not None and self.worker.running:
            # The worker thread is a daemon: closing the window ends it, and the server
            # notices within seconds that it stopped polling.
            self.worker.stop(wait=False)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()
