"""The window: a few settings, one big Start/Stop button, and a log of everything it does."""

from __future__ import annotations

import queue
import tkinter as tk
from tkinter import messagebox, ttk

import httpx

from fuzzy_worker.core import (
    IMAGE_BACKENDS,
    VERSION,
    Config,
    Worker,
    gpu_info,
    load_config,
    save_config,
)

PRIVACY = (
    "What this does: while it is running, Fuzzy's server can ask this computer to run one chat "
    "message or draw one picture, using the model servers YOU run here. It can't run programs, "
    "read your files or reach anything else on your network. Close this window (or press Stop) "
    "and it disconnects at once."
)


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(f"Fuzzy Worker {VERSION}")
        root.minsize(560, 640)
        self.config = load_config()
        self.worker: Worker | None = None
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self._build()
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(300, self._pump)
        if self.config.connect_on_start and self.config.token:
            root.after(500, self._toggle)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        c = self.config
        pad = {"padx": 10, "pady": 4}
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text=PRIVACY, wraplength=520, foreground="#555").pack(anchor="w", **pad)

        box = ttk.LabelFrame(frame, text="Connect")
        box.pack(fill="x", **pad)
        self.server = self._entry(box, "Server address", c.server_url or "https://")
        self.token = self._entry(box, "Token", c.token, show="*")

        share = ttk.LabelFrame(frame, text="What to share (only things running on THIS computer)")
        share.pack(fill="x", **pad)
        self.run_llm = tk.BooleanVar(value=c.run_llm)
        ttk.Checkbutton(
            share, text="Chat model (Ollama, LM Studio, llama.cpp...)", variable=self.run_llm
        ).pack(anchor="w", padx=8)
        self.llm_url = self._entry(share, "  Address", c.llm_url)
        self.llm_model = self._entry(share, "  Model name", c.llm_model)
        ttk.Label(share, text="Pictures:").pack(anchor="w", padx=8, pady=(8, 0))
        self.image_backend = tk.StringVar(value=c.image_backend)
        ttk.Combobox(
            share, textvariable=self.image_backend, values=IMAGE_BACKENDS, state="readonly"
        ).pack(anchor="w", padx=8)
        self.image_url = self._entry(share, "  Address", c.image_url)

        limits = ttk.LabelFrame(frame, text="Stay out of my way (0 = no limit)")
        limits.pack(fill="x", **pad)
        self.idle = self._entry(
            limits,
            "Only after this many minutes without keyboard or mouse",
            str(c.only_when_idle_minutes),
            width=6,
        )
        self.gpu_limit = self._entry(
            limits, "Pause if my GPU is busier than (%)", str(c.pause_above_gpu_percent), width=6
        )
        self.auto = tk.BooleanVar(value=c.connect_on_start)
        ttk.Checkbutton(limits, text="Connect when this program starts", variable=self.auto).pack(
            anchor="w", padx=8, pady=4
        )

        row = ttk.Frame(frame)
        row.pack(fill="x", **pad)
        self.button = ttk.Button(row, text="Start", command=self._toggle)
        self.button.pack(side="left")
        ttk.Button(row, text="Check my setup", command=self._check).pack(side="left", padx=8)
        self.status = ttk.Label(row, text="stopped")
        self.status.pack(side="left", padx=12)

        self.log = tk.Text(frame, height=10, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, **pad)

    def _entry(
        self, parent: tk.Misc, label: str, value: str, *, show: str = "", width: int = 48
    ) -> tk.StringVar:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=2)
        ttk.Label(row, text=label).pack(side="left")
        var = tk.StringVar(value=value)
        ttk.Entry(row, textvariable=var, show=show, width=width).pack(side="right")
        return var

    # ------------------------------------------------------------------ actions
    def _read(self) -> Config:
        c = self.config
        c.server_url = self.server.get().strip()
        c.token = self.token.get().strip()
        c.run_llm = self.run_llm.get()
        c.llm_url = self.llm_url.get().strip()
        c.llm_model = self.llm_model.get().strip()
        c.image_backend = self.image_backend.get()
        c.image_url = self.image_url.get().strip()
        c.only_when_idle_minutes = _whole(self.idle.get())
        c.pause_above_gpu_percent = _whole(self.gpu_limit.get())
        c.connect_on_start = self.auto.get()
        return c

    def _toggle(self) -> None:
        if self.worker is not None and self.worker.running:
            self._stop()
            return
        config = self._read()
        worker = Worker(
            config,
            on_status=lambda s: self.events.put(("status", s)),
            on_log=lambda s: self.events.put(("log", s)),
        )
        problem = worker.problem()
        if problem:
            messagebox.showwarning("Fuzzy Worker", problem)
            return
        save_config(config)
        self.worker = worker
        worker.start()
        self.button.configure(text="Stop")

    def _stop(self) -> None:
        if self.worker is not None:
            self.button.configure(state="disabled")
            self.root.update_idletasks()
            self.worker.stop()
            self.button.configure(state="normal", text="Start")

    def _check(self) -> None:
        c = self._read()
        lines = []
        gpu = gpu_info()
        lines.append(
            f"GPU: {gpu['gpu']} ({gpu['gpu_memory_mb']} MB)"
            if gpu
            else "GPU: none found (nvidia-smi)"
        )
        for label, on, url, path in (
            ("Chat model", c.run_llm, c.llm_url, "/models"),
            ("Picture server", c.image_backend != "none", c.image_url, "/"),
        ):
            if not on:
                continue
            try:
                r = httpx.get(url.rstrip("/") + path, timeout=5.0)
                lines.append(f"{label}: answers ({r.status_code}) at {url}")
            except httpx.HTTPError:
                lines.append(f"{label}: NOT answering at {url}. Is it running?")
        messagebox.showinfo("Fuzzy Worker", "\n".join(lines))

    def _pump(self) -> None:
        while True:
            try:
                kind, text = self.events.get_nowait()
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
            else:
                self.log.configure(state="normal")
                self.log.insert("end", text + "\n")
                self.log.see("end")
                self.log.configure(state="disabled")
        self.root.after(300, self._pump)

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
