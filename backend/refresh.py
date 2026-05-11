import subprocess
import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

STEPS = [
    ("Training price models",      [sys.executable, "backend/train_models.py"]),
    ("Training farm risk model",   [sys.executable, "backend/training_and_eval.py"]),
    ("Running forecast pipeline",  [sys.executable, "backend/run_pipeline.py"]),
]

def run_refresh():
    log.info(f"=== Refresh started at {datetime.utcnow()} ===")
    for name, cmd in STEPS:
        log.info(f"Starting: {name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(f"FAILED: {name}\n{result.stderr}")
            sys.exit(1)
        log.info(f"Done: {name}")
    log.info("=== Refresh complete ===")

if __name__ == "__main__":
    run_refresh()