#!/usr/bin/env python3
"""
PAN 2026 Voight-Kampff — AI Text Detection
Ensemble pipeline: ModernBERT + DeBERTa-v3 + RoBERTa + GPT-2 stats + XGBoost
"""

import os
import sys
import math
import re
import json
import warnings
import joblib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = Path(__file__).parent.resolve()

# ── Paths (relative to script location) ──────────────────────────
MODEL_PATHS = {
    "modernbert": str(SCRIPT_DIR / "models" / "modernbert"),
    "deberta":    str(SCRIPT_DIR / "models" / "deberta"),
    "roberta":    str(SCRIPT_DIR / "models" / "roberta"),
}
META_CLF  = str(SCRIPT_DIR / "models" / "meta" / "meta_classifier_v4.joblib")
SCALER    = str(SCRIPT_DIR / "models" / "meta" / "scaler_v4.joblib")

# ── Config ───────────────────────────────────────────────────────
CHUNK = {
    "modernbert": {"max_len": 8192, "stride": 0},
    "deberta":    {"max_len": 512,  "stride": 256},
    "roberta":    {"max_len": 512,  "stride": 256},
}

REJECT_LOW, REJECT_HIGH = 0.45, 0.55
DISAGREE_STD = 0.25
DISAGREE_LOW, DISAGREE_HIGH = 0.30, 0.70
SHORT_TEXT_MIN = 50
CLIP_LOW, CLIP_HIGH = 0.05, 0.95

FEATURE_COLUMNS = [
    "score_modernbert", "score_deberta", "score_roberta",
    "mean_logprob", "logprob_variance", "perplexity", "token_rank_avg",
    "sentence_ppl_variance", "sentence_ppl_burstiness",
    "entropy", "repetition_rate", "ttr",
    "avg_sent_len", "sent_len_std",
    "stopword_ratio", "punct_density", "cap_ratio",
    "text_len", "n_sentences", "n_words",
]

STOPWORDS = set(
    "the a an and or but in on at to for of with by from is was are "
    "were be been being have has had do does did will would could "
    "should may might shall can this that these those it its they "
    "them their we our you your he his she her i my me not no so "
    "if as than then when where who which what".split()
)


# ════════════════════════════════════════════════════════════════
#  PIPELINE
# ════════════════════════════════════════════════════════════════

class Pipeline:
    def __init__(self):
        self.tokenizers = {}
        self.models = {}
        for name, path in MODEL_PATHS.items():
            print(f"  Loading {name}...", flush=True)
            try:
                self.tokenizers[name] = AutoTokenizer.from_pretrained(path)
            except ValueError:
                self.tokenizers[name] = AutoTokenizer.from_pretrained(
                    path, use_fast=False
                )
            self.models[name] = (
                AutoModelForSequenceClassification
                .from_pretrained(path)
                .to(DEVICE)
                .eval()
            )

        print("  Loading GPT-2...", flush=True)
        self.gpt2_tok = AutoTokenizer.from_pretrained(str(SCRIPT_DIR / "models" / "gpt2"))
        self.gpt2_tok.pad_token = self.gpt2_tok.eos_token
        self.gpt2 = (
            AutoModelForCausalLM
            .from_pretrained(str(SCRIPT_DIR / "models" / "gpt2"))
            .to(DEVICE)
            .eval()        
        )

        self.meta_clf = joblib.load(META_CLF)
        self.scaler = joblib.load(SCALER)
        print("  Pipeline ready.", flush=True)

    @torch.no_grad()
    def _score_transformer(self, name, text):
        cfg = CHUNK[name]
        enc = self.tokenizers[name](
            text, truncation=True, max_length=cfg["max_len"],
            stride=cfg["stride"], return_overflowing_tokens=True,
            return_tensors="pt", padding=True,
        )
        enc.pop("overflow_to_sample_mapping", None)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = self.models[name](**enc).logits
        return float(F.softmax(logits, dim=-1)[:, 1].mean().item())

    @torch.no_grad()
    def _gpt2_features(self, text):
        enc = self.gpt2_tok(
            text, return_tensors="pt", truncation=True, max_length=1024
        ).to(DEVICE)
        if enc["input_ids"].shape[1] < 4:
            return {
                "mean_logprob": 0.0, "logprob_variance": 0.0,
                "perplexity": 100.0, "token_rank_avg": 50.0,
            }
        ids = enc["input_ids"]
        out = self.gpt2(**enc, labels=ids)
        logits = out.logits[..., :-1, :].contiguous()
        targets = ids[..., 1:].contiguous()
        lp = (
            F.log_softmax(logits, dim=-1)
            .gather(2, targets.unsqueeze(-1))
            .squeeze(-1)
            .cpu()
            .numpy()
            .flatten()
        )
        ranks = (
            torch.argsort(logits, dim=-1, descending=True)
            == targets.unsqueeze(-1)
        ).nonzero()[:, 2]
        return {
            "mean_logprob": float(lp.mean()),
            "logprob_variance": float(lp.var()),
            "perplexity": math.exp(min(-lp.mean(), 20)),
            "token_rank_avg": (
                float(ranks.float().mean()) if len(ranks) > 0 else 50.0
            ),
        }

    @torch.no_grad()
    def _sent_ppl(self, sents):
        ppls = []
        for s in sents[:30]:
            if len(s.split()) < 3:
                continue
            enc = self.gpt2_tok(
                s, return_tensors="pt", truncation=True, max_length=128
            ).to(DEVICE)
            if enc["input_ids"].shape[1] < 2:
                continue
            out = self.gpt2(**enc, labels=enc["input_ids"])
            ppls.append(math.exp(min(out.loss.item(), 20)))
        if len(ppls) < 2:
            return {"sentence_ppl_variance": 0.0, "sentence_ppl_burstiness": 0.0}
        a = np.array(ppls)
        return {
            "sentence_ppl_variance": float(np.var(a)),
            "sentence_ppl_burstiness": float(np.std(a) / (np.mean(a) + 1e-6)),
        }

    def _extract_all(self, text):
        f = {}
        for name in MODEL_PATHS:
            f[f"score_{name}"] = self._score_transformer(name, text)
        f.update(self._gpt2_features(text))
        sents = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text.strip())
            if len(s.strip()) > 5
        ]
        f.update(self._sent_ppl(sents))
        words = re.findall(r"\b\w+\b", text.lower())
        n = max(len(words), 1)
        slens = [len(s.split()) for s in sents] or [0]
        f["entropy"] = (
            -sum((c / n) * math.log2(c / n) for c in Counter(words).values())
            if n > 1 else 0.0
        )
        f["repetition_rate"] = (
            sum(
                c - 1
                for c in Counter(
                    tuple(words[i : i + 4]) for i in range(len(words) - 3)
                ).values()
            )
            / max(len(words) - 3, 1)
            if len(words) > 4 else 0.0
        )
        f["ttr"] = len(set(words)) / n
        f["avg_sent_len"] = float(np.mean(slens))
        f["sent_len_std"] = float(np.std(slens))
        f["stopword_ratio"] = sum(1 for w in words if w in STOPWORDS) / n
        f["punct_density"] = len(re.findall(r"[.,;:!?]", text)) / n
        f["cap_ratio"] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        f["text_len"] = float(len(text))
        f["n_sentences"] = float(len(sents))
        f["n_words"] = float(n)
        return f

    def _apply_rules(self, p_raw, scores, n_words):
        std = float(np.std(list(scores.values())))
        if REJECT_LOW <= p_raw <= REJECT_HIGH:
            return 0.5
        if std > DISAGREE_STD and DISAGREE_LOW < p_raw < DISAGREE_HIGH:
            return 0.5
        if n_words < SHORT_TEXT_MIN:
            return float(
                np.clip(0.5 + (p_raw - 0.5) * 0.5, CLIP_LOW, CLIP_HIGH)
            )
        return float(np.clip(p_raw, CLIP_LOW, CLIP_HIGH))

    def predict(self, text):
        if not text or len(text.strip()) < 5:
            return 0.5
        feats = self._extract_all(text)
        x = np.array(
            [feats.get(c, 0.0) for c in FEATURE_COLUMNS], dtype=np.float32
        ).reshape(1, -1)
        x_s = self.scaler.transform(x)
        p_raw = float(self.meta_clf.predict_proba(x_s)[0, 1])
        scores = {
            "modernbert": feats["score_modernbert"],
            "deberta": feats["score_deberta"],
            "roberta": feats["score_roberta"],
        }
        return self._apply_rules(p_raw, scores, int(feats["n_words"]))


# ════════════════════════════════════════════════════════════════
#  MAIN — PAN/TIRA entry point
# ════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_dir = sys.argv[2]
    else:
        input_file = os.environ.get("inputDataset", "")
        output_dir = os.environ.get("outputDir", "")
    
    if not input_file or not output_dir:
        print(f"Usage: {sys.argv[0]} <input.jsonl> <output_dir>")
        print("Or set env vars: inputDataset, outputDir")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("Initializing pipeline...", flush=True)
    pipeline = Pipeline()

    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Processing {len(records)} texts...", flush=True)
    output_path = os.path.join(output_dir, "predictions.jsonl")

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            if i % 50 == 0:
                print(f"  {i}/{len(records)}", flush=True)
            try:
                label = pipeline.predict(rec.get("text", ""))
            except Exception as e:
                print(f"  Error on {rec.get('id')}: {e}", flush=True)
                label = 0.5
            fout.write(
                json.dumps({"id": rec["id"], "label": round(label, 4)}) + "\n"
            )

    print(f"Done. Output: {output_path}", flush=True)


if __name__ == "__main__":
    main()