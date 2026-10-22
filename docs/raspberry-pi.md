# Using a Raspberry Pi 4 with Fuzzy

A Pi 4 can't run Fuzzy's models (no useful GPU, 2 to 8 GB of RAM) and shouldn't hold its memory or database.
It is well suited to the jobs a small always-on box does best, which are all things that stay up when the big
machine misbehaves. In order of usefulness:

1. **A watchdog and a physical kill switch** (included: `deploy/pi/fuzzy-pi.py`). It polls Fuzzy's `/health`,
   notifies you (log, an LED, and optionally an [ntfy](https://ntfy.sh) topic or any URL that accepts a POST) when
   Fuzzy stops answering or a GPU passes a temperature limit, and has an optional **push button** that pauses and
   resumes Fuzzy using the same kill switch as the Safety tab. A red LED shows it is paused. That is a real
   off-switch that doesn't depend on the website, the server's login, or the server being healthy.
2. **The public front door.** Run Caddy or a WireGuard/Tailscale endpoint on the Pi, so the GPU server itself is never
   exposed to the internet. The Pi forwards only to Fuzzy's port over the private link.
3. **Off-machine backups.** Attach a USB SSD and pull nightly copies of `data_dir` (it holds the audit log), a `pg_dump`, and the archive directory with `rsync` over SSH. If the server's disk
   dies, Fuzzy's memory survives.
4. **Later, a body for Fuzzy** (not built): a temperature/light/humidity sensor on the Pi could report readings to
   Fuzzy as real observations of its own physical environment, which it could notice and react to in its own way.
   That needs a small new input endpoint; ask if you want it.

## Setting up the watchdog and button

Wiring (BCM pin numbers, all optional): a push button between **GPIO 17** and ground; a green LED (with a resistor,
about 330 Ω) on **GPIO 27**; a red LED (same) on **GPIO 22**.

```bash
sudo mkdir -p /opt/fuzzy-pi && sudo cp deploy/pi/fuzzy-pi.py /opt/fuzzy-pi/
sudo cp deploy/pi/fuzzy-pi.service /etc/systemd/system/
sudo tee /etc/fuzzy-pi.env >/dev/null <<'ENV'
FUZZY_URL=http://<server-address>:8766
FUZZY_TOKEN=<contents of data_dir/api_token on the server>
BUTTON_PIN=17
GREEN_LED_PIN=27
RED_LED_PIN=22
NOTIFY_URL=https://ntfy.sh/<a-long-random-topic>
GPU_TEMP_LIMIT=85
ENV
sudo chmod 600 /etc/fuzzy-pi.env
sudo systemctl enable --now fuzzy-pi
journalctl -u fuzzy-pi -f
```

The operator token is a secret: keep it only in that root-only file, never in chat or in git. The compose file
publishes Fuzzy on `127.0.0.1` only, so to reach it from the Pi put the Pi's address in a reverse proxy, or change
the `ports:` mapping to the server's private LAN address (not a public one).
