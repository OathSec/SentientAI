"""``python -m fuzzy_worker`` opens the window; ``--headless`` runs from the saved settings."""

from __future__ import annotations

import sys
import time

from fuzzy_worker.core import Worker, load_config


def headless() -> int:
    config = load_config()
    worker = Worker(config, on_status=lambda s: print("status:", s), on_log=print)
    problem = worker.problem()
    if problem:
        print(problem)
        return 2
    worker.start()
    try:
        while worker.running:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()
    return 0


if __name__ == "__main__":
    if "--headless" in sys.argv:
        raise SystemExit(headless())
    from fuzzy_worker.ui import main

    main()
