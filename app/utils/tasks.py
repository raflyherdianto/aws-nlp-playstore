"""
Background task manager with progress tracking.

Runs long-running tasks (scraping, training) in daemon threads
and exposes progress via a simple in-memory dict polled by the frontend.
"""

import threading
import traceback
import logging

logger = logging.getLogger(__name__)

_tasks = {}
_lock = threading.Lock()


def create_task(task_id):
    """Register a new task with initial state."""
    with _lock:
        _tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'total': 100,
            'message': 'Memulai...',
            'result': None,
            'error': None,
        }


def update_progress(task_id, progress, total, message):
    """Update task progress (called from background thread)."""
    with _lock:
        if task_id in _tasks:
            _tasks[task_id].update({
                'progress': progress,
                'total': total,
                'message': message,
            })


def complete_task(task_id, result=None):
    """Mark task as completed with optional result dict."""
    with _lock:
        if task_id in _tasks:
            t = _tasks[task_id]
            t['status'] = 'completed'
            t['progress'] = t['total']
            t['message'] = 'Selesai!'
            t['result'] = result


def fail_task(task_id, error):
    """Mark task as failed with error message."""
    with _lock:
        if task_id in _tasks:
            _tasks[task_id]['status'] = 'failed'
            _tasks[task_id]['error'] = str(error)
            _tasks[task_id]['message'] = f'Error: {error}'


def get_task(task_id):
    """Get a snapshot of task state (thread-safe copy)."""
    with _lock:
        t = _tasks.get(task_id)
        return dict(t) if t else None


def run_in_background(task_id, app, func, **kwargs):
    """
    Run *func* in a daemon thread with Flask app context.

    func signature: func(task_id, **kwargs) → result dict
    On success, complete_task is called automatically.
    On failure, fail_task is called automatically.
    """
    def wrapper():
        with app.app_context():
            try:
                result = func(task_id, **kwargs)
                complete_task(task_id, result)
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}\n{traceback.format_exc()}")
                fail_task(task_id, str(e))

    t = threading.Thread(target=wrapper, daemon=True)
    t.start()
