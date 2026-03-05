"""
APScheduler-based cleanup for CSV, .joblib, and metrics files older than 60 minutes.
Runs every 5 minutes to check for expired files.
"""

import os
import time
import shutil
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler(daemon=True)


def cleanup_old_files(app):
    """Delete CSV, .joblib, and .json metric files older than 15 minutes."""
    with app.app_context():
        upload_folder = app.config['UPLOAD_FOLDER']
        charts_folder = os.path.join(app.static_folder, 'charts')
        now = time.time()
        max_age = 15 * 60  # 15 minutes in seconds

        # Clean upload files (CSV, model, metrics)
        if os.path.exists(upload_folder):
            for fname in os.listdir(upload_folder):
                fpath = os.path.join(upload_folder, fname)
                if os.path.isfile(fpath):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in ('.csv', '.joblib', '.json'):
                        file_age = now - os.path.getmtime(fpath)
                        if file_age > max_age:
                            try:
                                os.remove(fpath)
                                print(f"[Scheduler] Deleted expired file: {fname}")
                            except OSError as e:
                                print(f"[Scheduler] Error deleting {fname}: {e}")

        # Clean chart directories
        if os.path.exists(charts_folder):
            for dname in os.listdir(charts_folder):
                dpath = os.path.join(charts_folder, dname)
                if os.path.isdir(dpath):
                    dir_age = now - os.path.getmtime(dpath)
                    if dir_age > max_age:
                        try:
                            shutil.rmtree(dpath)
                            print(f"[Scheduler] Deleted expired chart dir: {dname}")
                        except OSError as e:
                            print(f"[Scheduler] Error deleting dir {dname}: {e}")


def init_scheduler(app):
    """Initialize the APScheduler with a 5-minute interval cleanup job."""
    if scheduler.running:
        return
    scheduler.add_job(
        func=cleanup_old_files,
        trigger='interval',
        minutes=5,
        args=[app],
        id='cleanup_old_files',
        replace_existing=True,
    )
    scheduler.start()
    print("[Scheduler] File cleanup scheduler started (15-min expiry, checked every 5 min)")
