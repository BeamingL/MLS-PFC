"""ReplayOrchestrator: top-level orchestrator linking scoring, memory buffer, and adaptive sampling."""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from .buffer import MemoryBuffer
from .decoder import selective_decode
from .sampler import PEGFSSampler
from .scorer import SkillScorer
from .utils import clone_lora_state_dict, extract_lora_parameters, safe_mean


logger = logging.getLogger(__name__)


class ReplayOrchestrator:
    """
    Closed-loop memory replay orchestrator.

    Paper module mapping:
    - Module A: 3D-DSRS -> SkillScorer
    - Module B: selective decode + MemoryBuffer
    - Module C: P-EGFS -> PEGFSSampler
    """

    def __init__(
        self,
        model,
        tokenizer,
        memory_capacity: int = 500,
        num_epochs: int = 5,
        replay_batch_size: int = 8,
        lambda_scale: float = 1.0,
        update_interval: int = 100,
        scorer_config: Dict[str, Any] | None = None,
        use_scoring: bool = True,
        use_adaptive_sampling: bool = True,
        use_capacity_balance: bool = True,
        train_batch_size: int = 4,
        learning_rate: float = 5e-5,
        seed: int = 42,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.memory_capacity = int(memory_capacity)
        self.num_epochs = int(num_epochs)
        self.replay_batch_size = int(replay_batch_size)
        self.use_scoring = bool(use_scoring)
        self.use_adaptive_sampling = bool(use_adaptive_sampling)
        self.use_capacity_balance = bool(use_capacity_balance)
        self.train_batch_size = int(train_batch_size)
        self.learning_rate = float(learning_rate)
        self._rng = random.Random(seed)

        scorer_config = scorer_config or {}
        self.scorer = SkillScorer(**scorer_config)
        self.memory_buffer = MemoryBuffer(
            total_capacity=self.memory_capacity,
            use_capacity_balance=self.use_capacity_balance,
        )
        self.sampler = PEGFSSampler(
            lambda_scale=lambda_scale,
            update_interval=update_interval,
            seed=seed,
            adaptive=self.use_adaptive_sampling,
        )

        # Task-level LoRA parameter archiving
        self.archived_lora_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.historical_params: Dict[int, torch.Tensor] = {}

    def train_task(self, task_id: int, train_dataset, eval_fn=None):
        """
        Train a single task: training + scoring + buffer insertion + parameter archiving.
        """
        task_id = int(task_id)
        records = self._dataset_to_records(train_dataset)
        if not records:
            raise ValueError("train_dataset is empty.")

        if self.use_scoring:
            self.scorer.init_task(num_samples=len(records))

        replay_enabled = self._is_replay_enabled()
        if replay_enabled and self.memory_buffer.num_tasks > 0:
            historical_params = {
                tid: self.historical_params[tid]
                for tid in self.memory_buffer.get_all_historical_samples().keys()
                if tid in self.historical_params
            }
            self.sampler.adaptive = self.use_adaptive_sampling
            self.sampler.init_for_new_task(
                memory_buffer=self.memory_buffer,
                historical_params=historical_params,
            )

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate,
        )

        global_step = 0
        device = getattr(self.model, "device", torch.device("cpu"))
        self.model.train()

        for epoch in range(1, self.num_epochs + 1):
            loader = DataLoader(
                records,
                batch_size=self.train_batch_size,
                shuffle=True,
                collate_fn=self._collate_train_batch,
            )
            epoch_sample_losses: Dict[int, List[float]] = {}

            for batch in loader:
                global_step += 1
                optimizer.zero_grad()

                main_outputs = self.model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                main_loss = self._extract_loss(main_outputs)
                total_loss = main_loss

                if replay_enabled and self.memory_buffer.num_tasks > 0:
                    if self.use_adaptive_sampling:
                        current_params = extract_lora_parameters(self.model)
                        self.sampler.maybe_update_probabilities(global_step, current_params)
                    replay_samples = self.sampler.sample_replay_batch(self.replay_batch_size)
                    if replay_samples:
                        replay_batch = self._build_replay_batch(replay_samples, device)
                        replay_outputs = self.model(**replay_batch)
                        replay_loss = self._extract_loss(replay_outputs)
                        total_loss = main_loss + replay_loss

                total_loss.backward()
                optimizer.step()

                if self.use_scoring:
                    batch_losses = self._compute_per_sample_losses(
                        logits=getattr(main_outputs, "logits", None),
                        labels=batch["labels"].to(device),
                        sample_indices=batch["sample_indices"],
                    )
                    for idx, val in batch_losses.items():
                        epoch_sample_losses.setdefault(idx, []).append(val)

            if self.use_scoring:
                avg_loss = {idx: safe_mean(vals) for idx, vals in epoch_sample_losses.items()}
                self.scorer.record_epoch_loss(epoch=epoch, sample_losses=avg_loss)
                self.scorer.update_scores(epoch=epoch)
                logger.info("Task %s Epoch %s Score distribution: %s", task_id, epoch, self.scorer.get_distribution())

        # Buffer insertion and strength update
        summary = self._finalize_task_memory(task_id=task_id, records=records)
        self._archive_task_lora(task_id=task_id)

        # Re-decode historical tasks and dynamically update S_i and M_final
        if replay_enabled and self.memory_buffer.num_tasks > 1:
            self._refresh_historical_scores(current_task_id=task_id)

        if eval_fn is not None:
            summary["eval"] = eval_fn(self.model, task_id)
        return summary

    def run_continual_learning(self, task_sequence: Sequence[Tuple[int, Any]]):
        """Run the full continual-learning sequence."""
        results = []
        for task_id, dataset in task_sequence:
            logger.info("Start task %s", task_id)
            result = self.train_task(task_id=task_id, train_dataset=dataset)
            results.append(result)
        return results

    def _is_replay_enabled(self) -> bool:
        # If all three switches are off, it degrades to Sequential Fine-tuning (no replay).
        return self.use_scoring or self.use_adaptive_sampling or self.use_capacity_balance

    def _finalize_task_memory(self, task_id: int, records: List[dict]) -> Dict[str, Any]:
        n_select = max(1, self.memory_capacity // max(1, self.memory_buffer.num_tasks + 1))
        if self.use_scoring:
            pool_indices = self.scorer.get_candidate_pool(n_select=n_select)
            candidates = [records[idx] for idx in pool_indices if 0 <= idx < len(records)]
        else:
            actual_n = min(n_select, len(records))
            candidates = self._rng.sample(records, actual_n)

        if not candidates:
            candidates = records[: min(len(records), n_select)]

        try:
            m_final_scores = selective_decode(
                model=self.model,
                tokenizer=self.tokenizer,
                samples=candidates,
                max_new_tokens=128,
            )
        except Exception as exc:
            logger.warning("selective_decode failed, fallback to zero score: %s", exc)
            m_final_scores = {str(s["sample_id"]): 0.0 for s in candidates}

        memory_strength = math.exp(safe_mean(m_final_scores.values(), default=0.0)) if m_final_scores else 1.0
        self.memory_buffer.update_task_buffer(
            task_id=task_id,
            candidates=candidates,
            m_final_scores=m_final_scores,
            memory_strength=memory_strength,
        )
        logger.info(
            "Task %s Inventory completed: candidates=%s strength=%.4f current_tasks=%s",
            task_id,
            len(candidates),
            memory_strength,
            self.memory_buffer.num_tasks,
        )
        return {
            "task_id": task_id,
            "num_candidates": len(candidates),
            "memory_strength": memory_strength,
            "memory_tasks": self.memory_buffer.num_tasks,
        }

    def _refresh_historical_scores(self, current_task_id: int) -> None:
        updated_m_final: Dict[int, Dict[str, float]] = {}
        new_strengths: Dict[int, float] = {}

        for task_id, samples in self.memory_buffer.get_all_historical_samples().items():
            if int(task_id) == int(current_task_id):
                continue
            if not samples:
                continue
            try:
                scores = selective_decode(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    samples=samples,
                    max_new_tokens=128,
                )
            except Exception as exc:
                logger.warning("Historical selective_decode failed for task %s: %s", task_id, exc)
                continue
            updated_m_final[int(task_id)] = scores
            new_strengths[int(task_id)] = math.exp(safe_mean(scores.values(), default=0.0))

        if updated_m_final:
            self.memory_buffer.rebalance(
                current_num_tasks=self.memory_buffer.num_tasks,
                updated_m_final=updated_m_final,
            )
        if new_strengths:
            self.memory_buffer.update_historical_strengths(new_strengths)

    def _archive_task_lora(self, task_id: int) -> None:
        self.archived_lora_states[int(task_id)] = clone_lora_state_dict(self.model)
        self.historical_params[int(task_id)] = extract_lora_parameters(self.model).detach().cpu()

    def _dataset_to_records(self, dataset) -> List[dict]:
        records = []
        for idx in range(len(dataset)):
            raw = dataset[idx]
            if not isinstance(raw, dict):
                raw = dict(raw)

            rec = dict(raw)
            rec["sample_id"] = str(rec.get("sample_id", idx))
            rec["_index"] = idx

            rec["input_ids"] = self._to_int_list(rec.get("input_ids", []))
            rec["labels"] = self._to_int_list(rec.get("labels", []))
            if "attention_mask" in rec:
                rec["attention_mask"] = self._to_int_list(rec["attention_mask"])
            else:
                rec["attention_mask"] = [1] * len(rec["input_ids"])

            if "input_text" not in rec:
                rec["input_text"] = self.tokenizer.decode(rec["input_ids"], skip_special_tokens=True)
            if "reference_text" not in rec:
                ref_ids = [x for x in rec["labels"] if x != -100]
                rec["reference_text"] = self.tokenizer.decode(ref_ids, skip_special_tokens=True)
            records.append(rec)
        return records

    def _collate_train_batch(self, batch: List[dict]) -> Dict[str, Any]:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        max_len = 1
        for rec in batch:
            max_len = max(max_len, len(rec["input_ids"]), len(rec["labels"]))

        input_ids = []
        attention_mask = []
        labels = []
        sample_indices = []
        for rec in batch:
            inp = rec["input_ids"] + [pad_token_id] * (max_len - len(rec["input_ids"]))
            att = rec["attention_mask"] + [0] * (max_len - len(rec["attention_mask"]))
            lab = rec["labels"] + [-100] * (max_len - len(rec["labels"]))
            input_ids.append(inp)
            attention_mask.append(att)
            labels.append(lab)
            sample_indices.append(int(rec["_index"]))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_indices": sample_indices,
        }

    def _build_replay_batch(self, replay_samples: List[dict], device: torch.device) -> Dict[str, torch.Tensor]:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        max_len = 1
        for rec in replay_samples:
            max_len = max(max_len, len(rec.get("input_ids", [])), len(rec.get("labels", [])))

        input_ids = []
        attention_mask = []
        labels = []
        for rec in replay_samples:
            inp = self._to_int_list(rec.get("input_ids", []))
            att = self._to_int_list(rec.get("attention_mask", [1] * len(inp)))
            lab = self._to_int_list(rec.get("labels", []))
            input_ids.append(inp + [pad_token_id] * (max_len - len(inp)))
            attention_mask.append(att + [0] * (max_len - len(att)))
            labels.append(lab + [-100] * (max_len - len(lab)))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
        }

    @staticmethod
    def _extract_loss(outputs):
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        if isinstance(outputs, (list, tuple)) and outputs:
            return outputs[0]
        raise ValueError("Model outputs do not contain loss.")

    @staticmethod
    def _compute_per_sample_losses(logits, labels: torch.Tensor, sample_indices: List[int]) -> Dict[int, float]:
        if logits is None:
            return {}
        if labels.size(1) < 2:
            return {}

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())
        mask = (shift_labels != -100).float()
        per_example = (token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)

        result = {}
        for i, sample_idx in enumerate(sample_indices):
            if i < per_example.size(0):
                result[int(sample_idx)] = float(per_example[i].detach().cpu().item())
        return result

    @staticmethod
    def _to_int_list(value: Any) -> List[int]:
        if isinstance(value, torch.Tensor):
            return [int(v) for v in value.detach().cpu().tolist()]
        if isinstance(value, list):
            return [int(v) for v in value]
        if isinstance(value, tuple):
            return [int(v) for v in value]
        return []
