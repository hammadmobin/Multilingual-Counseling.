"""Load HLQC (HighLowQualityCounseling) and PAIR, and select N client prompts by length."""
from __future__ import annotations

import csv
import io
import os
import re
import zipfile
from difflib import SequenceMatcher

import pandas as pd

TURN_RE = re.compile(r"^\s*([TC])\s*[:;]\s*(.*)$")  # "T;" typo exists in high_104


def norm(t) -> str:
    return (str(t).replace("\ufeff", "").replace("\u2019", "'").replace("\u2018", "'")
            .replace("\u201c", '"').replace("\u201d", '"'))


def _labels_from_csv(fh) -> dict:
    return {r["id"].strip(): r["label"].strip() for r in csv.DictReader(fh)}


def read_dataset(path: str) -> tuple[dict, dict]:
    """Return ({transcript_id: raw_text}, {transcript_id: label}). `path` = the .zip or any folder containing it."""
    files, labels = {}, {}
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if "__MACOSX" not in n and not n.endswith("/")]
            for n in names:
                if n.endswith("labels.csv"):
                    labels = _labels_from_csv(io.TextIOWrapper(z.open(n), encoding="utf-8-sig"))
                elif "/transcripts/" in n:
                    files[os.path.basename(n)] = z.read(n).decode("utf-8-sig", errors="replace")
    else:
        for root, _, fnames in os.walk(path):
            if "__MACOSX" in root:
                continue
            for f in fnames:
                p = os.path.join(root, f)
                if f == "labels.csv":
                    with open(p, encoding="utf-8-sig") as fh:
                        labels = _labels_from_csv(fh)
                elif os.path.basename(root) == "transcripts" and not f.startswith("."):
                    with open(p, encoding="utf-8-sig", errors="replace") as fh:
                        files[f] = fh.read()
    if not files:
        raise FileNotFoundError(f"No transcripts found under {path}")
    return files, labels


def parse_turns(text: str) -> list[tuple[str, str]]:
    """[(speaker, text)]; lines without a speaker tag are appended to the previous turn."""
    turns: list[list[str]] = []
    for line in norm(text).splitlines():
        if not line.strip():
            continue
        m = TURN_RE.match(line)
        if m:
            turns.append([m.group(1), m.group(2).strip()])
        elif turns:
            turns[-1][1] += " " + line.strip()
    return [(s, " ".join(t.split())) for s, t in turns]


def client_utterances(files: dict, labels: dict, quality: str = "high",
                      merge_consecutive: bool = True) -> pd.DataFrame:
    rows = []
    for tid in sorted(files):
        label = labels.get(tid) or tid.split("_")[0]
        if label != quality:
            continue
        turns = parse_turns(files[tid])
        if merge_consecutive:  # one client turn split over several C: lines -> one utterance
            merged = []
            for s, t in turns:
                if merged and s == "C" and merged[-1][0] == "C":
                    merged[-1] = ("C", merged[-1][1] + " " + t)
                else:
                    merged.append((s, t))
            turns = merged
        c_idx = 0
        for turn_idx, (s, t) in enumerate(turns):
            if s == "C" and t:
                rows.append({"transcript_id": tid, "turn_index": turn_idx,
                             "client_turn_index": c_idx, "prompt": t})
                c_idx += 1
    df = pd.DataFrame(rows)
    df["prompt_char_len"] = df["prompt"].str.len()
    df["prompt_word_len"] = df["prompt"].str.split().str.len()
    return df


def _near_dup(a: str, b: str, thr: float) -> bool:
    wa, wb = a.lower().split(), b.lower().split()
    sm = SequenceMatcher(None, wa, wb, autojunk=False)
    return sm.real_quick_ratio() >= thr and sm.quick_ratio() >= thr and sm.ratio() >= thr


def load_pair(path: str) -> pd.DataFrame:
    """PAIR csv: one client prompt per row (reflection columns hq1..lq5 are kept)."""
    df = pd.read_csv(path)
    df["prompt"] = df["prompt"].map(lambda t: " ".join(norm(t).split()))
    df.insert(0, "transcript_id", [f"pair_{i:03d}" for i in range(len(df))])
    df.insert(1, "turn_index", range(len(df)))
    df.insert(2, "client_turn_index", 0)
    df["prompt_char_len"] = df["prompt"].str.len()
    df["prompt_word_len"] = df["prompt"].str.split().str.len()
    return df


def select_prompts(df: pd.DataFrame, n: int = 20, length_by: str = "chars", select_by: str = "median",
                   dedup: bool = True, dedup_threshold: float = 0.8, max_per_transcript: int = 0) -> pd.DataFrame:
    """N prompts closest to the median / mean length (or the N longest). Dedup drops near-identical
    turns (the HLQC corpus has duplicate sessions, e.g. high_011 == high_014, high_017 ~ high_154)."""
    key = "prompt_char_len" if length_by == "chars" else "prompt_word_len"
    df = df.copy()
    if select_by == "longest":
        target = None
        df["_rank"] = -df[key]
    else:
        target = df[key].median() if select_by == "median" else df[key].mean()
        df["_rank"] = (df[key] - target).abs()
    ordered = df.sort_values(["_rank", "transcript_id", "turn_index"])
    picked, per_tid = [], {}
    for _, r in ordered.iterrows():
        if max_per_transcript and per_tid.get(r.transcript_id, 0) >= max_per_transcript:
            continue
        if dedup and any(_near_dup(r.prompt, p.prompt, dedup_threshold) for p in picked):
            continue
        picked.append(r)
        per_tid[r.transcript_id] = per_tid.get(r.transcript_id, 0) + 1
        if len(picked) == n:
            break
    out = pd.DataFrame(picked).drop(columns="_rank").reset_index(drop=True)
    out.insert(0, "utt_id", range(len(out)))
    out["length_target"] = target
    return out
