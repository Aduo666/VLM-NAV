from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from . import replay


class ReplayArgumentTests(unittest.TestCase):
    def test_batch_arguments_require_an_output_directory(self):
        cases = (
            ["dataset", "--batch"],
            ["dataset", "--batch", "--output-dir", "videos", "--output", "single.mp4"],
            ["episode_000001", "--output-dir", "videos"],
            ["episode_000001", "--overwrite"],
            ["episode_000001", "--fps", "nan"],
            ["episode_000001", "--fps", "0"],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with patch("sys.argv", ["replay", *arguments]), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        replay.parse_args()
                    self.assertEqual(raised.exception.code, 2)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "需要 FFmpeg 和 ffprobe")
class ReplayVideoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.output = self.root / "videos"

    def episode(self, scene="scene_a", name="episode_000001", *, fps=10., success=True):
        episode = self.dataset / scene / name
        for directory in ("front_rgb", "front_depth", "down_depth", "bev"):
            (episode / directory).mkdir(parents=True, exist_ok=True)
        count = 4
        pose = np.column_stack((np.linspace(0., .3, count), np.zeros((count, 2))))
        action = np.tile(np.array([.3, .1], dtype=np.float32), (count, 1))
        action[-1] = 0.
        np.savez_compressed(
            episode / "state.npz", sample_time_sec=np.arange(count) / fps,
            human_action_vw=action, applied_action_vw=action, pose_world_xyyaw=pose,
        )
        meta = {
            "episode_id": name.removeprefix("episode_"), "status": "complete",
            "success": success, "sample_count": count, "duration_sec": (count - 1) / fps,
            "start_pose_world_xyyaw": pose[0].tolist(),
            "goal_world_xy": pose[-1, :2].tolist() if success else None,
            "bev": {"forward_m": 3., "backward_m": 1., "left_m": 2., "right_m": 2.},
        }
        (episode / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        depth = np.full((48, 64), 2000, dtype=np.uint16)
        bev = np.zeros((3, 80, 80), dtype=np.uint8)
        bev[1] = 255
        bev[0, 15:20, 20:60] = 255
        bev[1, 15:20, 20:60] = 0
        for index in range(count):
            stem = f"{index:06d}"
            rgb = np.empty((48, 64, 3), dtype=np.uint8)
            rgb[:] = (30 + index * 30, 100 + index * 10, 180 - index * 15)
            self.assertTrue(cv2.imwrite(str(episode / "front_rgb" / f"{stem}.jpg"), rgb))
            for directory in ("front_depth", "down_depth"):
                self.assertTrue(cv2.imwrite(str(episode / directory / f"{stem}.png"), depth))
            np.save(episode / "bev" / f"{stem}.npy", bev)
        return episode

    def run_replay(self, *arguments):
        output = io.StringIO()
        with patch("sys.argv", ["replay", *map(str, arguments)]), redirect_stdout(output):
            code = replay.main()
        return code, output.getvalue()

    def probe(self, path):
        result = subprocess.run(
            [shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
             "-of", "json", str(path)],
            check=True, capture_output=True, text=True,
        )
        return json.loads(result.stdout)["streams"][0]

    def assert_video(self, path, *, codec="h264", fps=10.):
        stream = self.probe(path)
        self.assertEqual(stream["codec_name"], codec)
        self.assertEqual(int(stream["nb_read_frames"]), 4)
        self.assertEqual((stream["width"], stream["height"]), (1160, 360))
        self.assertAlmostEqual(float(Fraction(stream["avg_frame_rate"])), fps)
        if codec == "h264":
            self.assertEqual(stream["pix_fmt"], "yuv420p")

    def test_batch_preserves_scene_names_and_each_episode_fps(self):
        self.episode("scene_a", fps=10.)
        self.episode("scene_b", fps=5., success=False)
        self.episode("scene_a", ".episode_000002.tmp")
        code, output = self.run_replay(self.dataset, "--batch", "--output-dir", self.output)
        self.assertEqual(code, 0, output)
        self.assertIn("生成=2，跳过=0，失败=0", output)
        self.assert_video(self.output / "scene_a/episode_000001.mp4", fps=10.)
        self.assert_video(self.output / "scene_b/episode_000001.mp4", fps=5.)
        self.assertEqual(len(list(self.output.rglob("*.mp4"))), 2)
        reader = cv2.VideoCapture(str(self.output / "scene_a/episode_000001.mp4"))
        try:
            readable, frame = reader.read()
            self.assertTrue(readable)
            source = cv2.imread(str(self.dataset / "scene_a/episode_000001/front_rgb/000000.jpg"))
            np.testing.assert_allclose(frame[300, 240], source[20, 20], atol=6)
        finally:
            reader.release()

    def test_batch_accepts_one_scene_and_an_explicit_fps(self):
        episode = self.episode()
        code, output = self.run_replay(episode.parent, "--batch", "--output-dir", self.output, "--fps", 8)
        self.assertEqual(code, 0, output)
        self.assert_video(self.output / "scene_a/episode_000001.mp4", fps=8.)

    def test_single_episode_export_keeps_existing_command_and_codec(self):
        episode = self.episode()
        target = self.root / "single.mp4"
        code, output = self.run_replay(episode, "--output", target, "--fps", 8)
        self.assertEqual(code, 0, output)
        self.assert_video(target, codec="mpeg4", fps=8.)

    def test_existing_complete_videos_are_skipped_and_overwrite_regenerates(self):
        self.episode()
        arguments = (self.dataset, "--batch", "--output-dir", self.output)
        code, output = self.run_replay(*arguments)
        self.assertEqual(code, 0, output)
        target = self.output / "scene_a/episode_000001.mp4"
        before = target.stat().st_mtime_ns
        with patch.object(replay, "_export_h264", side_effect=AssertionError("完整视频应当跳过")):
            code, output = self.run_replay(*arguments, "--fps", 8)
        self.assertEqual(code, 0, output)
        self.assertIn("生成=0，跳过=1，失败=0", output)
        self.assertEqual(target.stat().st_mtime_ns, before)
        code, output = self.run_replay(*arguments, "--overwrite", "--fps", 8)
        self.assertEqual(code, 0, output)
        self.assertIn("生成=1，跳过=0，失败=0", output)
        self.assert_video(target, fps=8.)

    def test_truncated_video_is_regenerated(self):
        self.episode()
        arguments = (self.dataset, "--batch", "--output-dir", self.output)
        code, output = self.run_replay(*arguments)
        self.assertEqual(code, 0, output)
        target = self.output / "scene_a/episode_000001.mp4"
        contents = target.read_bytes()
        target.write_bytes(contents[:len(contents) // 2])
        self.assertFalse(replay._complete_h264(target, 4, shutil.which("ffprobe")))
        code, output = self.run_replay(*arguments)
        self.assertEqual(code, 0, output)
        self.assertIn("生成=1，跳过=0，失败=0", output)
        self.assert_video(target)

    def test_existing_mpeg4_video_is_regenerated_as_h264(self):
        episode = self.episode()
        target = self.output / "scene_a/episode_000001.mp4"
        code, output = self.run_replay(episode, "--output", target)
        self.assertEqual(code, 0, output)
        self.assert_video(target, codec="mpeg4")
        code, output = self.run_replay(self.dataset, "--batch", "--output-dir", self.output)
        self.assertEqual(code, 0, output)
        self.assertIn("生成=1，跳过=0，失败=0", output)
        self.assert_video(target)

    def test_failed_episode_preserves_its_video_and_later_episodes_continue(self):
        episode = self.episode()
        arguments = (self.dataset, "--batch", "--output-dir", self.output)
        code, output = self.run_replay(*arguments)
        self.assertEqual(code, 0, output)
        target = self.output / "scene_a/episode_000001.mp4"
        original = target.read_bytes()
        (episode / "front_rgb/000001.jpg").unlink()
        self.episode("scene_b")
        code, output = self.run_replay(*arguments, "--overwrite")
        self.assertEqual(code, 1, output)
        self.assertIn("生成=1，跳过=0，失败=1", output)
        self.assertEqual(target.read_bytes(), original)
        self.assert_video(self.output / "scene_b/episode_000001.mp4")
        self.assertEqual(len(list(self.output.rglob("*.mp4"))), 2)

    def test_interrupt_cleans_temporary_video_and_can_resume(self):
        self.episode()
        original_frames = replay._episode_frames

        def interrupted_frames(data):
            yield next(original_frames(data))
            raise KeyboardInterrupt

        arguments = (self.dataset, "--batch", "--output-dir", self.output)
        with patch.object(replay, "_episode_frames", interrupted_frames):
            code, output = self.run_replay(*arguments)
        self.assertEqual(code, 130, output)
        self.assertIn("已中断", output)
        self.assertEqual(list(self.output.rglob("*.mp4")), [])
        code, output = self.run_replay(*arguments)
        self.assertEqual(code, 0, output)
        self.assert_video(self.output / "scene_a/episode_000001.mp4")

    def test_missing_ffmpeg_is_reported_before_encoding(self):
        self.episode()
        with patch.object(replay.shutil, "which", return_value=None):
            code, output = self.run_replay(self.dataset, "--batch", "--output-dir", self.output)
        self.assertEqual(code, 1, output)
        self.assertIn("ffmpeg 和 ffprobe", output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
