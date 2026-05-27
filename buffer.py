"""Memory Buffer Manager: capacity allocation, insertion, pruning, and memory-strength management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BufferSample:
    sample_id: str
    payload: dict
    m_final: float


class MemoryBuffer:
    """Memory buffer manager."""

    def __init__(self, total_capacity: int, use_capacity_balance: bool = True):
        self.total_capacity = int(total_capacity)
        self.use_capacity_balance = bool(use_capacity_balance)
        self._task_samples: Dict[int, List[BufferSample]] = {}
        self._memory_strengths: Dict[int, float] = {}

    @property
    def num_tasks(self) -> int:
        """Number of currently stored tasks."""
        return len(self._task_samples)

    def get_task_samples(self, task_id: int) -> List[dict]:
        """Get buffered samples for a specific task."""
        samples = []
        for row in self._task_samples.get(int(task_id), []):
            merged = dict(row.payload)
            merged["sample_id"] = row.sample_id
            merged["m_final"] = row.m_final
            merged["task_id"] = int(task_id)
            samples.append(merged)
        return samples

    def get_all_historical_samples(self) -> Dict[int, List[dict]]:
        """Get all historical samples grouped by task."""
        return {task_id: self.get_task_samples(task_id) for task_id in sorted(self._task_samples)}

    def get_memory_strength(self, task_id: int) -> float:
        """Get memory strength S_i for a task."""
        return float(self._memory_strengths.get(int(task_id), 1.0))

    def update_task_buffer(self, task_id: int, candidates: List[dict], m_final_scores: Dict[str, float], memory_strength: float):
        """
        Insert a new task: keep top N_k samples ranked by M_final.
        If capacity balancing is enabled, trigger global rebalancing.
        """
        task_id = int(task_id)
        task_count = len(set(list(self._task_samples.keys()) + [task_id]))
        quota = self._capacity_per_task(task_count) if self.use_capacity_balance else self.total_capacity

        ranked = []
        for idx, sample in enumerate(candidates):
            sid = self._resolve_sample_id(sample, idx)
            ranked.append(
                BufferSample(
                    sample_id=sid,
                    payload=dict(sample),
                    m_final=float(m_final_scores.get(sid, 0.0)),
                )
            )
        ranked.sort(key=lambda x: x.m_final, reverse=True)
        self._task_samples[task_id] = ranked[: max(0, quota)]
        self._memory_strengths[task_id] = float(memory_strength)

        if self.use_capacity_balance:
            self.rebalance(current_num_tasks=task_count)

    def rebalance(self, current_num_tasks: int, updated_m_final: Dict[int, Dict[str, float]] | None = None):
        """
        Capacity rebalancing: shrink all tasks to floor(M/k).
        If updated_m_final is provided, update scores before sorting.
        """
        if not self.use_capacity_balance:
            return

        k = max(1, int(current_num_tasks))
        quota = self._capacity_per_task(k)
        for task_id, rows in self._task_samples.items():
            score_map = (updated_m_final or {}).get(task_id, {})
            if score_map:
                for row in rows:
                    if row.sample_id in score_map:
                        row.m_final = float(score_map[row.sample_id])
            rows.sort(key=lambda x: x.m_final, reverse=True)
            self._task_samples[task_id] = rows[:quota]

    def update_historical_strengths(self, new_strengths: Dict[int, float]):
        """Update memory strengths for historical tasks."""
        for task_id, val in new_strengths.items():
            self._memory_strengths[int(task_id)] = float(val)

    def _capacity_per_task(self, num_tasks: int) -> int:
        if num_tasks <= 0:
            return self.total_capacity
        return max(1, self.total_capacity // num_tasks)

    @staticmethod
    def _resolve_sample_id(sample: dict, fallback_idx: int) -> str:
        for key in ("sample_id", "id", "unique_id"):
            if key in sample:
                return str(sample[key])
        return str(fallback_idx)
