#!/usr/bin/env python3
"""A watchdog and a physical kill switch for Fuzzy, for a Raspberry Pi (or any small box).

* Watches Fuzzy's /health. If it stops answering, or a GPU runs hot, it tells you
  (a log line, an optional status LED, and an optional ntfy.sh/webhook notification).
* Optional hardware kill switch: a push button that pauses/resumes Fuzzy (the same kill
  switch as the Safety tab), with a red LED that lights while Fuzzy is paused.

Standard library only; GPIO uses gpiozero (preinstalled on Raspberry Pi OS) and is skipped when
no pins are configured. Configure with environment variables (see docs/raspberry-pi.md):

  FUZZY_URL        http://server:8765          where Fuzzy's site/API listens
  FUZZY_TOKEN      <operator token>            data_dir/api_token on the server (needed for the
                                               button and for GPU temperatures)
  BUTTON_PIN       17     push button to ground (BCM numbering): press to pause/resume
  GREEN_LED_PIN    27     lit while Fuzzy is up
  RED_LED_PIN      22     lit while Fuzzy is paused, blinks while it is unreachable
  NOTIFY_URL       https://ntfy.sh/your-secret-topic   POSTed a plain-text message on changes
  GPU_TEMP_LIMIT   85     alert above this many degrees C (needs FUZZY_TOKEN)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

URL = os.environ.get("FUZZY_URL", "http://127.0.0.1:8765").rstrip("/")
TOKEN = os.environ.get("FUZZY_TOKEN", "")
NOTIFY_URL = os.environ.get("NOTIFY_URL", "")
GPU_LIMIT = float(os.environ.get("GPU_TEMP_LIMIT", "85"))
EVERY_S = 20


def request(path: str, method: str = "GET", body: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    data = json.dumps(body).encode() if body is not None else (b"{}" if method == "POST" else None)
    req = urllib.request.Request(URL + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=8) as response:
        return json.loads(response.read() or b"{}")


def notify(message: str) -> None:
    print(time.strftime("%F %T"), message, flush=True)
    if not NOTIFY_URL:
        return
    try:
        urllib.request.urlopen(
            urllib.request.Request(NOTIFY_URL, data=message.encode(), method="POST"), timeout=8
        )
    except (urllib.error.URLError, OSError):
        pass


def pin(name: str):
    value = os.environ.get(name, "")
    return int(value) if value.isdigit() else None


def main() -> None:
    green = red = button = None
    if any(pin(n) is not None for n in ("BUTTON_PIN", "GREEN_LED_PIN", "RED_LED_PIN")):
        from gpiozero import LED, Button

        green = LED(pin("GREEN_LED_PIN")) if pin("GREEN_LED_PIN") is not None else None
        red = LED(pin("RED_LED_PIN")) if pin("RED_LED_PIN") is not None else None
        if pin("BUTTON_PIN") is not None:
            button = Button(pin("BUTTON_PIN"), pull_up=True, bounce_time=0.05)

    state = {"paused": False, "up": None, "hot": False}

    def toggle() -> None:
        try:
            if state["paused"]:
                request("/v1/resume", "POST")
                notify("Fuzzy resumed by the button")
            else:
                request("/v1/kill", "POST", {"reason": "pause button on the Raspberry Pi"})
                notify("Fuzzy paused by the button")
        except (urllib.error.URLError, OSError) as exc:
            notify(f"The button press didn't reach Fuzzy: {exc}")

    if button is not None:
        button.when_pressed = toggle

    failures = 0
    while True:
        try:
            health = request("/health")
            failures = 0
            state["paused"] = bool(health.get("kill_switch"))
            if state["up"] is False:
                notify("Fuzzy is reachable again")
            state["up"] = True
            if TOKEN:
                try:
                    gpus = request("/api/admin/hardware").get("gpus", [])
                    hottest = max((g["temp_c"] for g in gpus if g.get("temp_c") is not None), default=0)
                    if hottest >= GPU_LIMIT and not state["hot"]:
                        notify(f"A GPU is at {hottest:.0f} °C (limit {GPU_LIMIT:.0f})")
                    state["hot"] = hottest >= GPU_LIMIT
                except (urllib.error.URLError, OSError, ValueError):
                    pass  # temperatures are a bonus; /health is the point
        except (urllib.error.URLError, OSError, ValueError):
            failures += 1
            if failures == 3 and state["up"] is not False:
                notify(f"Fuzzy hasn't answered at {URL} for about a minute")
                state["up"] = False
        if green is not None:
            green.value = 1 if state["up"] and not state["paused"] else 0
        if red is not None:
            if state["up"] is False:
                red.blink(on_time=0.4, off_time=0.4)
            elif state["paused"]:
                red.on()
            else:
                red.off()
        time.sleep(EVERY_S)


if __name__ == "__main__":
    main()
