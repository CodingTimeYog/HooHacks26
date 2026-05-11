from apscheduler.schedulers.background import BackgroundScheduler
from backend.refresh import run_refresh
import time
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    
    # Run once immediately on startup
    log.info("Running initial refresh on startup...")
    run_refresh()

    # Then schedule daily at 6am UTC
    scheduler.add_job(run_refresh, "cron", hour=6, minute=0)
    scheduler.start()
    log.info("Scheduler running — refresh fires daily at 06:00 UTC")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()