"""Turn LLM outputs into the final table: one row per prompt."""
from __future__ import annotations

import pandas as pd

from .data import norm

ATTRS = ["category", "label", "intensity", "severity", "risk_type", "negated", "experiencer",
         "temporality", "hypothetical", "certainty", "mi_talk", "mi_talk_evidence"]


def cell(v):
    return "; ".join(v) if isinstance(v, list) else v


def build_table(name, sel, recs):
    rows = []
    for (_, r), rec in zip(sel.iterrows(), recs):
        o = rec["out"]
        tl = norm(r.prompt).lower()
        spans = [] if "error" in o else o.get("spans", [])
        pos = lambda s: tl.find(norm(s).lower())
        spans = sorted(spans, key=lambda s: (pos(s) < 0, pos(s)))        # order of appearance in the prompt
        rows.append({
            "dataset": name, "utt_id": r.utt_id, "transcript_id": r.transcript_id,
            "prompt": r.prompt,
            "spans": " | ".join(spans),
            **{a: cell(o.get(a, "")) for a in ATTRS},
            "core_meaning": o.get("core_meaning", ""),
            "n_spans": len(spans),
            "prompt_char_len": r.prompt_char_len, "prompt_word_len": r.prompt_word_len,
            "nonverbatim_spans": " | ".join(s for s in spans if pos(s) < 0),   # check these by hand
            "llm_error": o.get("error", ""),
        })
    return pd.DataFrame(rows)
