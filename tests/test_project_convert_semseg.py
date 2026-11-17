# -*- coding: utf-8 -*-
"""项目类型转换（含「实例分割 → 语义分割」）回归测试。

核心约定：
  - 平台标注统一是实例 polygon，seg / det / obb / semseg 共用同一份标注；
  - 转成 semseg 只改 projects.task_type，标注原样保留
    （训练时由 prepare_semantic_dataset 把实例合并成二值前景 mask）；
  - copy 模式必须把类别、图片文件、shape/shape_params 一起带过去；
  - cls（图级标签）不参与转换。

    .venv/bin/python -m unittest tests.test_project_convert_semseg -v
"""

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.config import settings
from server.database import Base
from server.models.annotation import Annotation
from server.models.defect_class import DefectClass
from server.models.image import Image
from server.models.project import Project
from server.services.project_convert import convert_project_task_type


class ProjectConvertSemsegTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        # _convert_copy 用 settings.upload_path 定位原图，临时指到 tmp
        self._old_root = settings.STORAGE_ROOT
        self._old_upload = settings.UPLOAD_DIR
        settings.STORAGE_ROOT = str(self.tmp)
        settings.UPLOAD_DIR = "uploads"

        self.engine = create_engine(f"sqlite:///{self.tmp / 'test.db'}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()

        # 源项目：实例分割，2 类，1 张图，2 个标注（1 多边形 + 1 参数化矩形）
        self.src = Project(name="src_seg", task_type="seg", crop_size=640, overlap=0.2)
        self.db.add(self.src)
        self.db.flush()
        self.c0 = DefectClass(project_id=self.src.id, class_index=0, name="隐裂", color="#FF4D4F")
        self.c1 = DefectClass(project_id=self.src.id, class_index=1, name="崩边", color="#45FF80")
        self.db.add_all([self.c0, self.c1])
        self.db.flush()

        img_dir = self.tmp / "uploads" / str(self.src.id)
        img_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / "abcd1234_a.png").write_bytes(b"fake-png-bytes")
        self.img = Image(
            project_id=self.src.id,
            filename="a.png",
            file_path=f"{self.src.id}/abcd1234_a.png",
            width=640,
            height=640,
            file_size=14,
            status="labeled",
            content_hash="0" * 64,
        )
        self.db.add(self.img)
        self.db.flush()

        self.db.add_all([
            Annotation(
                image_id=self.img.id, class_id=self.c0.id,
                polygon=[{"x": 0.1, "y": 0.1}, {"x": 0.4, "y": 0.1}, {"x": 0.4, "y": 0.4}],
                shape="polygon", area=100.0,
            ),
            Annotation(
                image_id=self.img.id, class_id=self.c1.id,
                polygon=[{"x": 0.5, "y": 0.5}, {"x": 0.7, "y": 0.5},
                         {"x": 0.7, "y": 0.8}, {"x": 0.5, "y": 0.8}],
                shape="rect",
                shape_params={"x1": 0.5, "y1": 0.5, "x2": 0.7, "y2": 0.8},
                area=200.0,
            ),
        ])
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()
        settings.STORAGE_ROOT = self._old_root
        settings.UPLOAD_DIR = self._old_upload
        self._tmp.cleanup()

    def _anns(self, project_id: int) -> list:
        return (
            self.db.query(Annotation)
            .join(Image, Annotation.image_id == Image.id)
            .filter(Image.project_id == project_id)
            .order_by(Annotation.id)
            .all()
        )

    def test_copy_seg_to_semseg(self):
        res = convert_project_task_type(self.src.id, "semseg", "copy", self.db)

        self.assertEqual(res["mode"], "copy")
        self.assertEqual(res["task_type"], "semseg")
        self.assertEqual(res["target_project_id"], res["source_project_id"] + 1)
        self.assertEqual(res["image_count"], 1)

        # 源项目完全不动
        self.db.refresh(self.src)
        self.assertEqual(self.src.task_type, "seg")
        self.assertEqual(len(self._anns(self.src.id)), 2)

        new_id = res["target_project_id"]
        new_proj = self.db.query(Project).filter(Project.id == new_id).first()
        self.assertEqual(new_proj.task_type, "semseg")

        # 类别按 class_index 复制（语义分割下类别只用于标注/统计，训练合并为前景）
        new_classes = (
            self.db.query(DefectClass)
            .filter(DefectClass.project_id == new_id)
            .order_by(DefectClass.class_index)
            .all()
        )
        self.assertEqual([c.class_index for c in new_classes], [0, 1])
        self.assertEqual([c.name for c in new_classes], ["隐裂", "崩边"])

        # 图片文件确实复制到新项目目录
        new_img = self.db.query(Image).filter(Image.project_id == new_id).first()
        self.assertTrue((settings.upload_path / new_img.file_path).exists())
        self.assertEqual(new_img.content_hash, "0" * 64)

        # 标注原样复制，参数化矩形不退化
        new_anns = self._anns(new_id)
        self.assertEqual(len(new_anns), 2)
        self.assertEqual(new_anns[0].shape, "polygon")
        self.assertEqual(new_anns[1].shape, "rect")
        self.assertAlmostEqual(new_anns[1].shape_params["x2"], 0.7)
        self.assertEqual(
            new_anns[1].class_id,
            next(c.id for c in new_classes if c.class_index == 1),
        )

        # 提示里必须说清“合并成二值前景”
        joined = " ".join(res["warnings"])
        self.assertIn("二值", joined)
        self.assertIn("合并", joined)

    def test_inplace_seg_to_semseg_keeps_annotations(self):
        before = [(a.id, a.class_id, a.shape, len(a.polygon)) for a in self._anns(self.src.id)]
        res = convert_project_task_type(self.src.id, "semseg", "inplace", self.db)

        self.assertEqual(res["mode"], "inplace")
        self.assertEqual(res["target_project_id"], self.src.id)
        self.assertEqual(res["annotation_count"], 2)
        self.db.refresh(self.src)
        self.assertEqual(self.src.task_type, "semseg")
        after = [(a.id, a.class_id, a.shape, len(a.polygon)) for a in self._anns(self.src.id)]
        self.assertEqual(before, after)
        self.assertTrue(res["warnings"])

    def test_semseg_back_to_seg(self):
        convert_project_task_type(self.src.id, "semseg", "inplace", self.db)
        res = convert_project_task_type(self.src.id, "seg", "inplace", self.db)
        self.assertEqual(res["task_type"], "seg")
        self.assertEqual(len(self._anns(self.src.id)), 2)
        self.assertTrue(any("实例多边形" in w for w in res["warnings"]))

    def test_cls_source_rejected(self):
        self.src.task_type = "cls"
        self.db.commit()
        with self.assertRaises(ValueError) as ctx:
            convert_project_task_type(self.src.id, "semseg", "copy", self.db)
        self.assertIn("cls", str(ctx.exception))

    def test_same_type_and_unknown_target_rejected(self):
        with self.assertRaises(ValueError):
            convert_project_task_type(self.src.id, "seg", "copy", self.db)
        with self.assertRaises(ValueError):
            convert_project_task_type(self.src.id, "yolo", "copy", self.db)

    def test_warns_when_no_annotations(self):
        self.db.query(Annotation).delete()
        self.db.commit()
        res = convert_project_task_type(self.src.id, "semseg", "inplace", self.db)
        self.assertTrue(any("没有任何多边形标注" in w for w in res["warnings"]))


if __name__ == "__main__":
    unittest.main()
