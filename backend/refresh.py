import subprocess
import sys
import logging
import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

STEPS = [
    ("Training price models",      [sys.executable, "backend/train_models.py"]),
    ("Running forecast pipeline",  [sys.executable, "backend/run_pipeline.py"]),
]

def run_refresh():
    log.info(f"=== Refresh started at {datetime.datetime.now(datetime.timezone.utc)} ===")    
    for name, cmd in STEPS:
        log.info(f"Starting: {name}")
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            log.error(f"FAILED: {name}")
            sys.exit(1)
        log.info(f"Done: {name}")
    log.info("=== Refresh complete ===")

if __name__ == "__main__":
    run_refresh()