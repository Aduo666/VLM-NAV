from __future__ import annotations

import logging
import multiprocessing
from multiprocessing import reduction
import os
import select
import signal
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from constraint_traverse.lifecycle import BackgroundTasks, ThreadWorker, bind_shutdown, exit_with_parent
from constraint_traverse.navigation import SamplingNavigationController, _SamplingJob
from constraint_traverse.benchmark.run import load_config
from constraint_traverse.benchmark.model import STATE_SIZE
from constraint_traverse.test_navigation import ROOT, make_grid
from vln_nav.app import navigation
from vln_nav.navigation.trajectory_clearance import FootprintConfig, FootprintPartConfig
from vln_nav.runtime.agx_profile import load_agx_profile


def guarded_child(ready):
    exit_with_parent()
    ready.send(True)
    ready.close()
    threading.Event().wait()


def parent_fixture(control):
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    child = context.Process(target=guarded_child, args=(writer,), daemon=True)
    child.start()
    writer.close()
    reader.recv()
    reader.close()
    control.send(child.pid)
    reduction.send_handle(control, child.sentinel, os.getppid())
    command = control.recv()
    if command == "exit":
        os._exit(0)
    threading.Event().wait()


def uncooperative_child(ready):
    ready.send(True)
    ready.close()
    threading.Event().wait()


def create_controller():
    path = ROOT / "constraint_traverse/configs/benchmark.yaml"
    config = load_config(path)
    config["planner"]["device"] = "cpu"
    config["planner"]["samples"] = 16
    hardware = load_agx_profile(project_root=ROOT).agx_controller.continuous
    footprint = FootprintConfig((FootprintPartConfig("body", -0.15, 0.15, -0.1, 0.1),), 0.03)
    with patch("constraint_traverse.navigation.load_config", return_value=config):
        return SamplingNavigationController(method="MPPI", hardware=hardware, footprint=footprint,
            physics_dt=1 / 60, config_path=path, logger=logging.getLogger("shutdown-test"))


class LifecycleTests(unittest.TestCase):
    def test_all_tasks_close_even_when_one_fails(self):
        tasks = BackgroundTasks(Mock())
        first, broken, last = Mock(), Mock(), Mock()
        broken.close.side_effect = RuntimeError("test")
        for task in (first, broken, last):
            tasks.add(task)
        tasks.close()
        tasks.close()
        for task in (first, broken, last):
            task.close.assert_called_once_with(wait=True)

    def test_shutdown_event_closes_tasks(self):
        tasks, resource = BackgroundTasks(Mock()), Mock()
        tasks.add(resource)
        application = Mock()
        subscription = bind_shutdown(application, 1, tasks)
        method = application.get_shutdown_event_stream.return_value.create_subscription_to_pop_by_type
        self.assertIs(subscription, method.return_value)
        event_type, callback = method.call_args.args
        self.assertEqual(event_type, 1)
        callback(object())
        resource.close.assert_called_once_with(wait=True)

    def test_closed_group_closes_new_resource(self):
        tasks = BackgroundTasks(Mock())
        tasks.close()
        task = Mock()
        tasks.add(task)
        task.close.assert_called_once_with(wait=True)

    def test_closed_worker_rejects_jobs(self):
        worker = ThreadWorker()
        worker.close(wait=True)
        self.assertFalse(worker.submit(object()))
        self.assertIsNone(worker.poll())

    def test_navdp_close_prevents_followup_inference(self):
        started, release = threading.Event(), threading.Event()
        client = Mock()
        client.reset.side_effect = lambda intrinsic: (started.set(), release.wait(2))
        worker = navigation._NavDPWorker(client)
        job = types.SimpleNamespace(intrinsics=np.eye(3))
        self.assertTrue(worker.submit(job))
        self.assertTrue(started.wait(2))
        worker.close()
        release.set()
        worker.close(wait=True)
        self.assertFalse(worker._thread.is_alive())
        client.pointgoal_step.assert_not_called()

    def test_fine_search_cancels_on_close(self):
        entered = threading.Event()
        planner = Mock()
        planner.select_local_route.return_value = ((1., 0.), np.asarray([[0, 0], [1, 0]]))

        def search(**kwargs):
            entered.set()
            while not kwargs["cancel_requested"]():
                time.sleep(0.005)
            raise RuntimeError("cancelled")

        planner.plan.side_effect = search
        worker = navigation._FinePlanningWorker(planner=planner, restricted_analyzer=None, trajectory_refiner=Mock())
        job = navigation._FinePlanningJob(1, make_grid(), (0, 0, 0), (1, 0), None, False, 1.2, 0, 0)
        with patch.object(navigation, "clear_current_footprint", side_effect=lambda grid, **kwargs: grid):
            worker.submit(job)
            self.assertTrue(entered.wait(2))
            worker.close(wait=True)
        self.assertFalse(worker._thread.is_alive())
        self.assertFalse(worker.submit(job))

    def test_busy_process_is_reaped_and_close_is_idempotent(self):
        controller = create_controller()
        job = _SamplingJob(controller._epoch, np.zeros(STATE_SIZE, np.float32), make_grid(),
            np.asarray([1, 0], np.float32), 0, None, np.empty((0, 2), np.float32), None)
        controller._worker.submit(job)
        pid = controller._worker.process.pid
        controller.close(wait=True)
        controller.close(wait=True)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertFalse(controller._worker.submit(job))

    def test_unresponsive_owned_process_is_terminated(self):
        context = multiprocessing.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        controller = create_controller()
        process = context.Process(target=uncooperative_child, args=(writer,), daemon=True)
        process.start()
        writer.close()
        self.assertTrue(reader.poll(10))
        reader.recv()
        reader.close()
        pid = process.pid
        controller._worker.process = process
        started = time.monotonic()
        controller.close(wait=False)
        self.assertLess(time.monotonic() - started, 2.5)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    @unittest.skipUnless(os.name == "posix", "需要 Unix 进程句柄传递")
    def test_child_exits_on_abrupt_parent_exit_and_kill(self):
        context = multiprocessing.get_context("spawn")
        for abrupt in (True, False):
            with self.subTest(abrupt=abrupt):
                reader, writer = context.Pipe()
                parent = context.Process(target=parent_fixture, args=(writer,))
                parent.start()
                writer.close()
                child_pid = None
                descriptor = None
                try:
                    self.assertTrue(reader.poll(15))
                    child_pid = reader.recv()
                    descriptor = reduction.recv_handle(reader)
                    if abrupt:
                        reader.send("exit")
                    else:
                        parent.kill()
                    parent.join(timeout=5)
                    self.assertFalse(parent.is_alive())
                    self.assertTrue(select.select([descriptor], [], [], 5)[0])
                finally:
                    if parent.is_alive():
                        parent.kill()
                        parent.join(timeout=2)
                    if descriptor is not None:
                        if child_pid is not None and not select.select([descriptor], [], [], 0)[0]:
                            try:
                                os.kill(child_pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        os.close(descriptor)
                    reader.close()
                    parent.close()


if __name__ == "__main__":
    unittest.main()
