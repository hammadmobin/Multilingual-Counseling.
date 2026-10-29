"""Must-preserve extraction, LLM only, one row per prompt.

  python run.py --dry-run          # only select prompts and print them (no API calls)
  python run.py --limit 5          # test on 5 prompts per dataset
  python run.py                    # full run: 20 prompts per dataset
"""
from __future__ import annotations

import argparse
import os

from tqdm.auto import tqdm

from mustpreserve.data import client_utterances, load_pair, read_dataset, select_prompts
from mustpreserve.llm import LLMExtractor, sys_for
from mustpreserve.table import build_table


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hlqc", default="data/HighLowQualityCounseling.zip", help="HLQC zip or extracted folder")
    p.add_argument("--pair", default="data/pair_data_eng.csv", help="PAIR csv")
    p.add_argument("--datasets", nargs="+", default=["hlqc", "pair"], choices=["hlqc", "pair"])
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--quality", default="high", choices=["high", "low"], help="HLQC sessions to use")
    p.add_argument("--n", type=int, default=20, help="prompts per dataset")
    p.add_argument("--select-by", default="median", choices=["median", "mean", "longest"])
    p.add_argument("--length-by", default="chars", choices=["chars", "words"])
    p.add_argument("--max-per-transcript", type=int, default=0, help="0 = no cap")
    p.add_argument("--no-dedup", action="store_true", help="keep near-duplicate turns")
    p.add_argument("--no-merge-consecutive", action="store_true", help="don't join consecutive C: lines")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (optional)")
    p.add_argument("--limit", type=int, default=0, help="run only the first K prompts per dataset")
    p.add_argument("--dry-run", action="store_true", help="select prompts only, no API calls")
    return p.parse_args()


def main():
    a = parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    except ImportError:
        pass
    os.makedirs(a.out_dir, exist_ok=True)
    paths = {"hlqc": a.hlqc, "pair": a.pair}
    key = "prompt_char_len" if a.length_by == "chars" else "prompt_word_len"

    sels = {}
    for name in a.datasets:
        path = paths[name]
        if path.lower().endswith(".csv"):
            utts = load_pair(path)
        else:
            files, labels = read_dataset(path)
            utts = client_utterances(files, labels, a.quality, merge_consecutive=not a.no_merge_consecutive)
        sel = select_prompts(utts, a.n, a.length_by, a.select_by, dedup=not a.no_dedup,
                             max_per_transcript=a.max_per_transcript)
        sels[name] = sel.head(a.limit) if a.limit else sel
        target = sel.length_target[0]
        print(f"{name}: {len(utts)} client utterances | "
              f"{a.select_by} {a.length_by} = {'-' if target is None else round(target, 1)} | "
              f"selected {len(sel)} prompts, {a.length_by} {sel[key].min()}-{sel[key].max()}")
        if a.dry_run:
            for _, r in sel.iterrows():
                print(f"  [{r.utt_id:2d}] {r.transcript_id:10s} {r[key]:4d}  {r.prompt[:90]}")

    if a.dry_run:
        return

    llm = LLMExtractor(a.model, os.path.join(a.out_dir, "llm_cache.jsonl"), a.base_url)
    for name, sel in sels.items():
        sys_prompt = sys_for(paths[name])
        recs = [llm.extract(t, sys_prompt) for t in tqdm(sel["prompt"], desc=name)]
        table = build_table(name, sel, recs)
        out = os.path.join(a.out_dir, f"{name}_mustpreserve.csv")
        table.to_csv(out, index=False)
        print(f"{name}: saved {out} {table.shape} | spans {table.n_spans.sum()} | "
              f"prompts with non-verbatim spans {(table.nonverbatim_spans != '').sum()} | "
              f"LLM errors {(table.llm_error != '').sum()} | "
              f"tokens in/out {sum(r['in_tok'] for r in recs)}/{sum(r['out_tok'] for r in recs)}")


if __name__ == "__main__":
    main()
