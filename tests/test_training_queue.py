# -*- coding: utf-8 -*-
"""训练队列（TrainingDispatcher）行为回归测试。

只测调度器的排队逻辑，不跑真实训练：把 `run_training_pipeline` 换成
一个会记录调用顺序并 sleep 的假函数，同时绕过出队前的查库检查。

    .venv/bin/python -m unittest tests.test_training_queue -v
"""

import threading
import time
import unittest

from server.tasks import train_task as tt


class _FakePipeline:
    def __init__(self, delay: float = 0.15):
        self.delay = delay
        self.ran: list[int] = []
        self._lock = threading.Lock()

    def __call__(self, task_id: int) -> None:
        with self._lock:
            self.ran.append(task_id)
        time.sleep(self.delay)


class TrainingQueueTest(unittest.TestCase):
    def setUp(self):
        self.fake = _FakePipeline()
        self._orig_pipeline = tt.run_training_pipeline
        self._orig_skip = tt.TrainingDispatcher._skip_reason
        tt.run_training_pipeline = self.fake
        tt.TrainingDispatcher._skip_reason = staticmethod(lambda _tid: None)
        self.dispatcher = tt.TrainingDispatcher()

    def tearDown(self):
        # 等本用例的排队/运行线程全部收尾，否则遗留线程会拿下一个用例的假流水线
        # （模块级函数已被换掉）乱往 ran 里追加，造成跨用例污染
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not self.dispatcher.is_running and not self.dispatcher.queue_snapshot():
                break
            time.sleep(0.02)
        tt.run_training_pipeline = self._orig_pipeline
        tt.TrainingDispatcher._skip_reason = self._orig_skip

    def _wait_until(self, cond, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if cond():
                return True
            time.sleep(0.02)
        return cond()

    def test_first_runs_rest_queue_fifo(self):
        self.assertTrue(self.dispatcher.start(1))     # 立刻开跑
        self.assertFalse(self.dispatcher.start(2))    # 排队
        self.assertFalse(self.dispatcher.start(3))    # 排队
        self.assertEqual(self.dispatcher.queue_snapshot(), [2, 3])
        self.assertEqual(self.dispatcher.queue_position(2), 1)
        self.assertEqual(self.dispatcher.queue_position(3), 2)
        self.assertIsNone(self.dispatcher.queue_position(1))

        # 1 跑完后 2 自动开跑，队列只剩 3
        self.assertTrue(self._wait_until(lambda: self.fake.ran[:2] == [1, 2]))
        self.assertTrue(self._wait_until(lambda: self.dispatcher.queue_snapshot() == [3]))
        # 最终三个都按顺序跑完
        self.assertTrue(self._wait_until(lambda: self.fake.ran == [1, 2, 3]))

    def test_duplicate_start_not_queued_twice(self):
        self.assertTrue(self.dispatcher.start(1))
        self.assertFalse(self.dispatcher.start(2))
        self.assertFalse(self.dispatcher.start(2))
        self.assertEqual(self.dispatcher.queue_snapshot(), [2])

    def test_drop_queued_removes_task(self):
        self.assertTrue(self.dispatcher.start(1))
        self.assertFalse(self.dispatcher.start(2))
        self.assertFalse(self.dispatcher.start(3))
        self.assertTrue(self.dispatcher.drop_queued(2))
        self.assertFalse(self.dispatcher.drop_queued(2))     # 已经不在队列里
        self.assertEqual(self.dispatcher.queue_snapshot(), [3])
        self.assertTrue(self._wait_until(lambda: self.fake.ran == [1, 3]))

    def test_queue_runs_after_cancelled_current(self):
        self.assertTrue(self.dispatcher.start(1))
        self.assertFalse(self.dispatcher.start(2))
        # 当前任务被取消（状态变更）也照样轮到下一个
        self.assertTrue(self._wait_until(lambda: self.fake.ran == [1, 2]))


if __name__ == "__main__":
    unittest.main()
