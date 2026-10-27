# -*- coding: utf-8 -*-
"""存储分层（原图/归档在机械盘、训练工作区在 SSD）回归测试。

只测路径解析与归档搬运，不碰真实训练：

    .venv/bin/python -m unittest tests.test_storage_layout -v
"""

import tempfile
import unittest
from pathlib import Path

from server.config import settings
from server.services import artifact_cleanup as ac


class StorageLayoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._orig = (settings.STORAGE_ROOT, settings.RUNS_DIR, settings.ARCHIVE_DIR)
        settings.STORAGE_ROOT = str(root / "storage")
        settings.RUNS_DIR = str(root / "ssd" / "runs")
        settings.ARCHIVE_DIR = str(root / "hdd" / "archive")
        self.root = root

    def tearDown(self):
        settings.STORAGE_ROOT, settings.RUNS_DIR, settings.ARCHIVE_DIR = self._orig
        self.tmp.cleanup()

    def test_absolute_dir_used_as_is(self):
        self.assertEqual(settings.runs_path, (self.root / "ssd" / "runs").resolve())
        self.assertTrue(settings.runs_path.is_dir())
        self.assertTrue(settings.archive_path.is_dir())

    def test_relative_dir_falls_back_to_storage_root(self):
        settings.RUNS_DIR = "runs"
        self.assertEqual(settings.runs_path, Path(settings.STORAGE_ROOT).resolve() / "runs")

    def test_archive_moves_task_dir_and_is_idempotent(self):
        src = settings.runs_path / "task_1"
        (src / "runs" / "train" / "weights").mkdir(parents=True)
        (src / "runs" / "train" / "weights" / "best.pt").write_bytes(b"w")

        moved = ac.archive_task(1, str(src))
        self.assertIsNotNone(moved)
        new_dir, old_dir = moved
        self.assertEqual(new_dir, settings.archive_path / "task_1")
        self.assertFalse(old_dir.exists())
        self.assertTrue((new_dir / "runs" / "train" / "weights" / "best.pt").is_file())

        # 已经在归档盘：原样返回，不重复搬
        again = ac.archive_task(1, str(new_dir))
        self.assertEqual(again, (new_dir, new_dir))

    def test_archive_never_overwrites_existing_dir(self):
        src = settings.runs_path / "task_2"
        src.mkdir(parents=True)
        (src / "marker.txt").write_text("new")
        existing = settings.archive_path / "task_2"
        existing.mkdir(parents=True)
        (existing / "marker.txt").write_text("old")

        moved = ac.archive_task(2, str(src))
        self.assertIsNotNone(moved)
        new_dir, _ = moved
        self.assertNotEqual(new_dir, existing)
        self.assertEqual((existing / "marker.txt").read_text(), "old")
        self.assertEqual((new_dir / "marker.txt").read_text(), "new")

    def test_sweep_orphan_archive_skips_db_backups(self):
        root = settings.archive_path
        (root / "task_9").mkdir(parents=True, exist_ok=True)
        (root / "task_10_20260101-000000").mkdir(parents=True, exist_ok=True)
        (root / "db_backups").mkdir(parents=True, exist_ok=True)
        (root / "db_backups" / "app-1.db").write_bytes(b"x")

        removed = ac.sweep_orphan_archive_dirs([9])
        self.assertEqual(removed, ["task_10_20260101-000000"])
        self.assertTrue((root / "db_backups" / "app-1.db").exists())
        self.assertTrue((root / "task_9").exists())

    def test_url_to_disk_resolves_runs_parent_root(self):
        infer = settings.runs_path / "inference" / "a.png"
        infer.parent.mkdir(parents=True, exist_ok=True)
        infer.write_bytes(b"x")
        self.assertEqual(
            ac.url_to_disk("/static/storage/runs/inference/a.png"), infer.resolve()
        )


if __name__ == "__main__":
    unittest.main()
