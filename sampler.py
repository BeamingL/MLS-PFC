"""P-EGFS 采样器：基于参数漂移与记忆强度的自适应回放。"""

from __future__ import annotations

import math
import random
from typing import Dict, List

import torch

from .buffer import MemoryBuffer
from .utils import ShuffledCycleIterator


class PEGFSSampler:
    """参数化记忆曲线自适应回放采样器。"""

    def __init__(self, lambda_scale: float = 1.0, update_interval: int = 100, seed: int = 42, adaptive: bool = True):
        self.lambda_scale = float(lambda_scale)
        self.update_interval = int(update_interval)
        self.adaptive = bool(adaptive)
        self._rng = random.Random(seed)

        self._memory_buffer: MemoryBuffer | None = None
        self._historical_params: Dict[int, torch.Tensor] = {}
        self._task_ids: List[int] = []
        self._probabilities: Dict[int, float] = {}
        self._iterators: Dict[int, ShuffledCycleIterator] = {}

    def init_for_new_task(self, memory_buffer: MemoryBuffer, historical_params: Dict[int, torch.Tensor]):
        """训练新任务前，初始化任务迭代器与初始概率。"""
        self._memory_buffer = memory_buffer
        self._historical_params = {int(k): v.detach().float().cpu() for k, v in historical_params.items()}
        self._task_ids = []
        self._iterators = {}

        for task_id, samples in memory_buffer.get_all_historical_samples().items():
            if samples:
                self._task_ids.append(int(task_id))
                self._iterators[int(task_id)] = ShuffledCycleIterator(list(samples), self._rng)

        self._task_ids.sort()
        if not self._task_ids:
            self._probabilities = {}
            return
        uniform = 1.0 / len(self._task_ids)
        self._probabilities = {tid: uniform for tid in self._task_ids}

    def maybe_update_probabilities(self, step: int, current_params: torch.Tensor):
        """每隔 U 步更新一次回放概率。"""
        if not self._task_ids:
            return
        if self.update_interval <= 0:
            return
        if int(step) % self.update_interval != 0:
            return

        if not self.adaptive:
            uniform = 1.0 / len(self._task_ids)
            self._probabilities = {tid: uniform for tid in self._task_ids}
            return

        cur = current_params.detach().float().cpu()
        forgetting: Dict[int, float] = {}
        for task_id in self._task_ids:
            theta_i = self._historical_params.get(task_id)
            if theta_i is None or theta_i.numel() == 0:
                forgetting[task_id] = 0.0
                continue

            min_len = min(cur.numel(), theta_i.numel())
            cur_slice = cur[:min_len]
            hist_slice = theta_i[:min_len]
            denominator = torch.linalg.vector_norm(hist_slice, ord=2).item()
            if denominator <= 1e-12:
                denominator = 1e-12
            t_i = torch.linalg.vector_norm(cur_slice - hist_slice, ord=2).item() / denominator

            strength = self._memory_buffer.get_memory_strength(task_id) if self._memory_buffer else 1.0
            strength = max(float(strength), 1e-12)
            f_i = 1.0 - math.exp(-(self.lambda_scale * t_i) / strength)
            forgetting[task_id] = max(0.0, float(f_i))

        total = sum(forgetting.values())
        if total <= 1e-12:
            uniform = 1.0 / len(self._task_ids)
            self._probabilities = {tid: uniform for tid in self._task_ids}
            return
        self._probabilities = {tid: forgetting[tid] / total for tid in self._task_ids}

    def sample_replay_batch(self, batch_size: int) -> List[dict]:
        """按当前概率分布采样一批回放样本。"""
        if batch_size <= 0 or not self._task_ids:
            return []

        weights = [self._probabilities.get(tid, 0.0) for tid in self._task_ids]
        if sum(weights) <= 1e-12:
            weights = [1.0 / len(self._task_ids) for _ in self._task_ids]

        batch = []
        for _ in range(batch_size):
            task_id = self._rng.choices(self._task_ids, weights=weights, k=1)[0]
            iterator = self._iterators.get(task_id)
            if iterator is None:
                continue
            sample = dict(iterator.next())
            sample["task_id"] = task_id
            batch.append(sample)
        return batch

    def get_current_probabilities(self) -> Dict[int, float]:
        """返回当前采样概率。"""
        return dict(self._probabilities)
