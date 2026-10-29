# Must-preserve extraction for MI counseling (LLM only)

Step 1 of a culture-adapted Spanish translation pipeline: for each client (patient) utterance, an LLM lists
everything the translation **must not lose or distort** — the verbatim spans plus 12 whole-prompt attributes.

- **HLQC** (HighLowQualityCounseling): high-quality sessions, client (`C:`) turns only.
- **PAIR**: the `prompt` column.
- From each dataset, **20 prompts closest to the median length** (`--select-by mean|longest` also available).
  Near-duplicate turns (the HLQC corpus contains duplicated sessions) are dropped.
- One LLM call per prompt (default `gpt-4o-mini`, OpenAI Structured Outputs). Calls are cached in
  `outputs/llm_cache.jsonl`, so re-runs are free.

## Setup
```bash
git clone <your-repo-url> && cd mustpreserve-llm
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # add your OpenAI key
```
Put `HighLowQualityCounseling.zip` and `pair_data_eng.csv` in `data/`.

## Run
```bash
python run.py --dry-run      # see the selected prompts, no API calls
python run.py --limit 5      # test on 5 prompts per dataset
python run.py                # full run (20 + 20 prompts)
```
Options: `--n`, `--select-by median|mean|longest`, `--length-by chars|words`, `--datasets hlqc pair`,
`--max-per-transcript`, `--model`, `--base-url`, `--hlqc PATH`, `--pair PATH` (see `python run.py -h`).

The same pipeline as a Colab notebook: `notebooks/step1_mustpreserve_prompt_level.ipynb`.

## Output
`outputs/hlqc_mustpreserve.csv` and `outputs/pair_mustpreserve.csv`, one row per prompt:

| # | Column | Content |
|---|---|---|
| 1–3 | `dataset`, `utt_id`, `transcript_id` | identifiers |
| 4 | `prompt` | the client utterance |
| 5 | `spans` | all must-preserve spans, separated by ` \| ` |
| 6 | `category` | all present: symptom, health_condition, emotion, life_event, target_behavior, risk, quantity |
| 7 | `label` | main topic, 2–5 words |
| 8 | `intensity` | low / moderate / high / unspecified |
| 9 | `severity` | low / moderate / high |
| 10 | `risk_type` | all risk types present (incl. denied), or `not_risk` |
| 11 | `negated` | True if any key meaning is negated |
| 12 | `experiencer` | everyone described, e.g. `client; partner` |
| 13 | `temporality` | current / past / future / habitual / mixed / unspecified |
| 14 | `hypothetical` | True / False |
| 15 | `certainty` | certain / hedged / mixed |
| 16 | `mi_talk` | change / sustain / ambivalent / neutral |
| 17 | `mi_talk_evidence` | verbatim span showing the MI talk |
| 18 | `core_meaning` | one sentence the translation must convey |
| 19–21 | `n_spans`, `prompt_char_len`, `prompt_word_len` | descriptive |
| 22 | `nonverbatim_spans` | spans not found verbatim in the prompt (check by hand) |
| 23 | `llm_error` | empty unless the API call failed |

Columns 6–17 are the 12 attributes.

## Layout
```
run.py                      CLI
mustpreserve/data.py        load HLQC / PAIR, select prompts
mustpreserve/llm.py         system prompt, JSON schema, cached OpenAI extractor
mustpreserve/table.py       build the one-row-per-prompt table
notebooks/                  Colab version
data/                       datasets (git-ignored)
```
# Multilingual-Counseling.
