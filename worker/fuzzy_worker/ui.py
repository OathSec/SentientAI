"""The window. Three tabs: Home (connect, start/stop), Set up this PC (one click), Limits & log."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from fuzzy_worker.core import (
    IMAGE_BACKENDS,
    VERSION,
    Config,
    Worker,
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

PRIVACY = (
    "While this is running, Fuzzy's server can ask this computer to run one chat message or "
    "draw one picture, using the model on THIS computer. It can't run programs, read your "
    "files or reach anything else on your network. Press Stop (or close the window) and it "
    "disconnects at once."
)
MODELS = ("qwen3:4b", "qwen3:8b", "qwen3:14b")


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(f"Fuzzy Worker {VERSION}")
        root.geometry("620x700")
        root.minsize(560, 620)
        self.config = load_config()
        self.worker: Worker | None = None
        self.events: queue.Queue[tuple[str, str, float | None]] = queue.Queue()
        self.setting_up = False
        self.gpu = gpu_info()
        self._build()
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(300, self._pump)
        if self.config.connect_on_start and self.config.token and self.config.run_llm:
            root.after(800, self._toggle)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        tabs = ttk.Notebook(self.root)
        tabs.pack(fill="both", expand=True, padx=8, pady=8)
        home, setup, more = (
            ttk.Frame(tabs, padding=12),
            ttk.Frame(tabs, padding=12),
            ttk.Frame(tabs, padding=12),
        )
        tabs.add(home, text="  Home  ")
        tabs.add(setup, text="  Set up this PC  ")
        tabs.add(more, text="  Limits & log  ")
        self.tabs = tabs
        self._home(home)
        self._setup_tab(setup)
        self._more(more)
        if not self.config.run_llm:
            tabs.select(setup)  # first time: start with the setup

    def _home(self, f: ttk.Frame) -> None:
        c = self.config
        ttk.Label(f, text="Lend this computer to Fuzzy", font=("Segoe UI", 15, "bold")).pack(
            anchor="w"
        )
        ttk.Label(f, text=PRIVACY, wraplength=540, foreground="#555").pack(anchor="w", pady=(4, 10))

        box = ttk.LabelFrame(f, text="1. Connect", padding=8)
        box.pack(fill="x", pady=4)
        ttk.Label(box, text="Paste the connection code you were given:").pack(anchor="w")
        self.code = tk.StringVar()
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Entry(row, textvariable=self.code).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Use it", command=self._use_code).pack(side="left", padx=6)
        self.saved = ttk.Label(box, text="", foreground="#2a7")
        self.saved.pack(anchor="w")
        adv = ttk.Frame(box)
        self.server = tk.StringVar(value=c.server_url)
        self.token = tk.StringVar(value=c.token)
        self._entry(adv, "Or the address", self.server)
        self._entry(adv, "and the token", self.token, show="*")
        self._show_adv = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            box,
            text="Enter address and token by hand",
            variable=self._show_adv,
            command=lambda: adv.pack(fill="x") if self._show_adv.get() else adv.pack_forget(),
        ).pack(anchor="w")
        self._refresh_saved()

        run = ttk.LabelFrame(f, text="2. Run", padding=8)
        run.pack(fill="x", pady=8)
        self.button = ttk.Button(run, text="Start", command=self._toggle)
        self.button.pack(side="left")
        self.status = ttk.Label(run, text="stopped", font=("Segoe UI", 10, "bold"))
        self.status.pack(side="left", padx=14)

        self.auto = tk.BooleanVar(value=c.connect_on_start)
        self.winstart = tk.BooleanVar(value=c.start_with_windows)
        ttk.Checkbutton(
            f, text="Connect automatically when the program starts", variable=self.auto
        ).pack(anchor="w")
        ttk.Checkbutton(
            f, text="Start the program when Windows starts", variable=self.winstart
        ).pack(anchor="w")
        self.summary = ttk.Label(f, text="", foreground="#555", wraplength=540)
        self.summary.pack(anchor="w", pady=10)
        self._refresh_summary()

    def _setup_tab(self, f: ttk.Frame) -> None:
        c = self.config
        ttk.Label(f, text="Get this PC ready", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        tag, why = recommend(self.gpu.get("gpu_memory_mb"))
        ttk.Label(f, text=f"Graphics card: {describe(self.gpu)}").pack(anchor="w", pady=(6, 0))
        ttk.Label(f, text=why, foreground="#555", wraplength=540).pack(anchor="w")
        ttk.Label(
            f,
            text="One click installs what's missing (Ollama, which runs the model), downloads the "
            "model (several GB), tunes it for Fuzzy and tests it. It can take a while; the "
            "program keeps working in the background.",
            wraplength=540,
        ).pack(anchor="w", pady=10)
        row = ttk.Frame(f)
        row.pack(anchor="w")
        ttk.Label(row, text="Model:").pack(side="left")
        self.model = tk.StringVar(value=tag)
        ttk.Combobox(row, textvariable=self.model, values=MODELS, state="readonly", width=12).pack(
            side="left", padx=6
        )
        self.setup_button = ttk.Button(f, text="Set everything up", command=self._setup)
        self.setup_button.pack(anchor="w", pady=10)
        self.bar = ttk.Progressbar(f, length=540, maximum=100)
        self.bar.pack(anchor="w")
        self.step = ttk.Label(
            f, text="Ollama: " + ("installed" if find_ollama() else "not installed yet")
        )
        self.step.pack(anchor="w", pady=4)
        ttk.Separator(f).pack(fill="x", pady=12)
        ttk.Label(
            f, text="Pictures (optional, for the more technical)", font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")
        ttk.Label(
            f,
            text="To draw pictures too, install Stable Diffusion WebUI (started with --api) and "
            "pick it here.",
            wraplength=540,
            foreground="#555",
        ).pack(anchor="w")
        self.image_backend = tk.StringVar(value=c.image_backend)
        ttk.Combobox(
            f, textvariable=self.image_backend, values=IMAGE_BACKENDS, state="readonly", width=16
        ).pack(anchor="w", pady=4)
        self.image_url = tk.StringVar(value=c.image_url)
        self._entry(f, "Picture server address", self.image_url)

    def _more(self, f: ttk.Frame) -> None:
        c = self.config
        ttk.Label(f, text="Stay out of my way (0 = no limit)", font=("Segoe UI", 10, "bold")).pack(
            anchor="w"
        )
        self.idle = tk.StringVar(value=str(c.only_when_idle_minutes))
        self.gpu_limit = tk.StringVar(value=str(c.pause_above_gpu_percent))
        self._entry(f, "Only after this many minutes without keyboard or mouse", self.idle, width=6)
        self._entry(f, "Pause if my graphics card is busier than (%)", self.gpu_limit, width=6)
        ttk.Label(f, text="Log", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(14, 2))
        self.log = tk.Text(f, height=16, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True)

    def _entry(
        self, parent: tk.Misc, label: str, var: tk.StringVar, *, show: str = "", width: int = 40
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label).pack(side="left")
        ttk.Entry(row, textvariable=var, show=show, width=width).pack(side="right")

    # ------------------------------------------------------------------ actions
    def _refresh_saved(self) -> None:
        ok = bool(self.server.get() and self.token.get())
        self.saved.configure(text="Connected details saved." if ok else "Nothing saved yet.")

    def _refresh_summary(self) -> None:
        c = self.config
        parts = []
        parts.append(
            f"Chat model: {c.llm_model}"
            if c.run_llm and c.llm_model
            else "Chat model: not set up yet (see the next tab)"
        )
        parts.append("Pictures: on" if c.image_backend != "none" else "Pictures: off")
        self.summary.configure(text="   |   ".join(parts))

    def _use_code(self) -> None:
        parsed = parse_code(self.code.get())
        if parsed is None:
            messagebox.showwarning("Fuzzy Worker", "That doesn't look like a connection code.")
            return
        self.server.set(parsed[0])
        self.token.set(parsed[1])
        self.code.set("")
        self._refresh_saved()

    def _read(self) -> Config:
        c = self.config
        c.server_url = self.server.get().strip()
        c.token = self.token.get().strip()
        c.image_backend = self.image_backend.get()
        c.image_url = self.image_url.get().strip()
        c.only_when_idle_minutes = _whole(self.idle.get())
        c.pause_above_gpu_percent = _whole(self.gpu_limit.get())
        c.connect_on_start = self.auto.get()
        c.start_with_windows = self.winstart.get()
        return c

    def _toggle(self) -> None:
        if self.worker is not None and self.worker.running:
            self.button.configure(state="disabled")
            self.root.update_idletasks()
            self.worker.stop()
            self.button.configure(state="normal", text="Start")
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
        self.button.configure(text="Stop")

    def _setup(self) -> None:
        if self.setting_up:
            return
        if find_ollama() is None and not messagebox.askyesno(
            "Fuzzy Worker",
            "Ollama (the program that runs the model) isn't installed. Download it from "
            "ollama.com and install it for you now?",
        ):
            return
        self.setting_up = True
        self.setup_button.configure(state="disabled")
        config = self._read()
        tag = self.model.get()

        def work() -> None:
            try:
                result = run_setup(
                    config,
                    lambda text, fraction: self.events.put(("step", text, fraction)),
                    model=tag,
                )
                save_config(config)
                self.events.put(("done", result, 1.0))
            except SetupError as exc:
                self.events.put(("fail", str(exc), None))
            except Exception as exc:  # shown, never a crash
                self.events.put(
                    ("fail", f"Something went wrong ({type(exc).__name__}: {exc})", None)
                )

        threading.Thread(target=work, name="setup", daemon=True).start()

    def _pump(self) -> None:
        while True:
            try:
                kind, text, fraction = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status.configure(text=text)
                if (
                    text.startswith("stopped")
                    and self.worker is not None
                    and not self.worker.running
                ):
                    self.button.configure(text="Start")
            elif kind == "step":
                self.step.configure(text=text)
                self.bar.configure(mode="determinate" if fraction is not None else "indeterminate")
                if fraction is None:
                    self.bar.start(15)
                else:
                    self.bar.stop()
                    self.bar["value"] = fraction * 100
                self._write_log(text) if fraction in (None, 0.0, 1.0) else None
            elif kind in ("done", "fail"):
                self.bar.stop()
                self.setting_up = False
                self.setup_button.configure(state="normal")
                self.step.configure(text=text if kind == "fail" else "Ready. " + text)
                self._write_log(text)
                if kind == "done":
                    self.bar["value"] = 100
                    self._refresh_summary()
                    self.tabs.select(0)
                    messagebox.showinfo(
                        "Fuzzy Worker",
                        "This PC is set up. Connect it on the Home tab and press Start.",
                    )
                else:
                    messagebox.showwarning("Fuzzy Worker", text)
            else:
                self._write_log(text)
        self.root.after(300, self._pump)

    def _write_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _close(self) -> None:
        if self.worker is not None and self.worker.running:
            self.worker.stop()
        self.root.destroy()


def _whole(text: str) -> int:
    try:
        return max(0, int(text.strip() or 0))
    except ValueError:
        return 0


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()
