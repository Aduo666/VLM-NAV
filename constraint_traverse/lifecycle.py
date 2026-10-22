from __future__ import annotations

import atexit
import multiprocessing
import os
import queue
import threading


def exit_with_parent():
    parent = multiprocessing.parent_process()
    if parent is not None:
        def watch_parent():
            parent.join()
            os._exit(0)
        threading.Thread(target=watch_parent, name="navigation-parent-watch", daemon=True).start()


def bind_shutdown(application, event_type, tasks):
    return application.get_shutdown_event_stream().create_subscription_to_pop_by_type(
        event_type, lambda event: tasks.close(), name="navigation-background-cleanup")


class BackgroundTasks:
    def __init__(self, logger):
        self.logger = logger
        self.closed = False
        self.tasks = []
        self.lock = threading.Lock()
        atexit.register(self.close)

    def add(self, task):
        with self.lock:
            closed = self.closed
            if not closed:
                self.tasks.append(task)
        if closed:
            task.close(wait=True)
        return task

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            tasks, self.tasks = self.tasks[::-1], []
        atexit.unregister(self.close)
        for task in tasks:
            try:
                task.close(wait=True)
            except Exception:
                self.logger.exception("[NavigationShutdown] 后台任务清理失败")


class ThreadWorker:
    def __init__(self):
        self._busy = False
        self._closed = False
        self._lock = threading.Lock()
        self._results = queue.Queue()
        self._thread = None
        self._stop_requested = threading.Event()

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def submit(self, job):
        with self._lock:
            if self._busy or self._closed:
                return False
            self._busy = True
            self._thread = threading.Thread(target=self._run, args=(job,), daemon=True)
            self._thread.start()
        return True

    def poll(self):
        if self._closed:
            return None
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def close(self, wait=False):
        with self._lock:
            self._closed = True
            self._stop_requested.set()
            thread = self._thread
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
