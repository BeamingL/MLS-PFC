"""3D-DSRS: 三维动态技能评分模块。"""

from __future__ import annotations

from typing import Dict, List

from .utils import percentile_25


class SkillScorer:
    """
    三维动态技能评分系统 (3D-DSRS)。

    论文对应关系：
    - 代理学习速度 v_t(d)
    - 代理稳定性 S_t(d)
    - 离散评分 r_t(d) in {1,2,3,4,5}
    """

    def __init__(self, tau_stable: float = 0.7, tau_unstable: float = 0.3, alpha: float = 0.1, epsilon: float = 1e-8):
        self.tau_stable = float(tau_stable)
        self.tau_unstable = float(tau_unstable)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self._scores: Dict[int, int] = {}
        self._loss_history: Dict[int, Dict[int, float]] = {}

    def init_task(self, num_samples: int):
        """新任务开始时初始化评分与 loss 记录。"""
        self._scores = {idx: 3 for idx in range(int(num_samples))}
        self._loss_history = {idx: {} for idx in range(int(num_samples))}

    def record_epoch_loss(self, epoch: int, sample_losses: Dict[int, float]):
        """记录每个 epoch 的样本 loss。"""
        epoch = int(epoch)
        for sample_id, loss in sample_losses.items():
            sid = int(sample_id)
            if sid not in self._scores:
                self._scores[sid] = 3
                self._loss_history[sid] = {}
            self._loss_history[sid][epoch] = float(loss)

    def update_scores(self, epoch: int):
        """epoch>=2 时按 3D-DSRS 规则更新评分。"""
        epoch = int(epoch)
        if epoch < 2:
            return

        current_losses = []
        for sid in self._scores:
            if epoch in self._loss_history.get(sid, {}):
                current_losses.append(self._loss_history[sid][epoch])
        p25 = percentile_25(current_losses)

        for sid, score in list(self._scores.items()):
            history = self._loss_history.get(sid, {})
            if epoch not in history or (epoch - 1) not in history:
                continue

            l_t = history[epoch]
            l_prev = history[epoch - 1]
            v_t = l_prev - l_t

            if 1 in history:
                l_1 = history[1]
            else:
                continue
            abs_sum = 0.0
            for k in range(2, epoch + 1):
                if (k - 1) in history and k in history:
                    abs_sum += abs(history[k - 1] - history[k])
            s_t = (l_1 - l_t) / (abs_sum + self.epsilon)

            step = self._rating_step(v_t=v_t, l_t=l_t, p25=p25, s_t=s_t)
            self._scores[sid] = int(max(1, min(5, score + step)))

    def _rating_step(self, v_t: float, l_t: float, p25: float, s_t: float) -> int:
        """评分步进函数 φ_t(d)。"""
        if ((v_t > 0.0) or (l_t < p25)) and (s_t >= self.tau_stable):
            return 1
        if (v_t < -self.alpha) or (s_t <= self.tau_unstable):
            return -1
        return 0

    def get_candidate_pool(self, n_select: int) -> List[int]:
        """
        按评分高到低构建候选池，返回索引列表。
        为满足 |C|>=n_select，会保留到“临界分层”全部样本。
        """
        if n_select <= 0:
            return []
        ranked = sorted(
            self._scores.keys(),
            key=lambda sid: (
                -self._scores[sid],
                self._loss_history.get(sid, {}).get(max(self._loss_history.get(sid, {}) or [0]), float("inf")),
                sid,
            ),
        )
        if len(ranked) <= n_select:
            return ranked
        threshold_score = self._scores[ranked[n_select - 1]]
        return [sid for sid in ranked if self._scores[sid] >= threshold_score]

    def get_scores(self) -> Dict[int, int]:
        """返回当前全部样本评分。"""
        return dict(self._scores)

    def get_distribution(self) -> Dict[int, int]:
        """返回评分分布，便于日志统计。"""
        dist = {i: 0 for i in range(1, 6)}
        for s in self._scores.values():
            dist[int(s)] = dist.get(int(s), 0) + 1
        return dist
