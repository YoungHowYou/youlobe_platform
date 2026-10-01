# -*- coding: utf-8 -*-
"""
后台训练任务包（无 Celery/Redis）
==================================
训练流水线在 uvicorn 进程内的守护线程中执行，单并发。
入口见 train_task.run_training_pipeline 与 training_dispatcher。
"""
