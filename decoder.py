"""选择性解码模块：生成 + chrF/ROUGE-L/BLEU-4 综合评分。"""

from __future__ import annotations

from typing import Dict, List

import torch

from .utils import min_max_normalize


def selective_decode(model, tokenizer, samples: List[dict], max_new_tokens: int = 128) -> Dict[str, float]:
    """
    对候选样本做自回归生成并计算 M_final。

    M_final(d) = (Norm(chrF) + Norm(ROUGE-L) + Norm(BLEU-4)) / 3
    """
    if not samples:
        return {}

    try:
        from rouge_score import rouge_scorer
        from sacrebleu.metrics import BLEU, CHRF
    except Exception as exc:  # pragma: no cover - 依赖缺失时明确提示
        raise ImportError("selective_decode 需要安装 `sacrebleu` 和 `rouge-score`。") from exc

    sample_ids = [str(s.get("sample_id", idx)) for idx, s in enumerate(samples)]
    prompts = [str(s.get("input_text", "")) for s in samples]
    references = [str(s.get("reference_text", "")) for s in samples]

    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    device = getattr(model, "device", torch.device("cpu"))
    if isinstance(encoded, dict):
        encoded = {k: v.to(device) for k, v in encoded.items()}
    else:
        encoded = encoded.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "pad_token_id", 0) or getattr(tokenizer, "eos_token_id", 0),
        )

    prompt_len = encoded["input_ids"].shape[1]
    generated_only = outputs[:, prompt_len:]
    hypotheses = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

    chrf_metric = CHRF()
    bleu_metric = BLEU(effective_order=True)
    rouge_metric = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    chrf_scores: Dict[str, float] = {}
    rouge_scores: Dict[str, float] = {}
    bleu_scores: Dict[str, float] = {}

    for sid, ref, hyp in zip(sample_ids, references, hypotheses):
        chrf_scores[sid] = float(chrf_metric.sentence_score(hyp, [ref]).score)
        bleu_scores[sid] = float(bleu_metric.sentence_score(hyp, [ref]).score)
        rouge_scores[sid] = float(rouge_metric.score(ref, hyp)["rougeL"].fmeasure)

    norm_chrf = min_max_normalize(chrf_scores)
    norm_rouge = min_max_normalize(rouge_scores)
    norm_bleu = min_max_normalize(bleu_scores)

    final_scores: Dict[str, float] = {}
    for sid in sample_ids:
        final_scores[sid] = (norm_chrf[sid] + norm_rouge[sid] + norm_bleu[sid]) / 3.0
    return final_scores
