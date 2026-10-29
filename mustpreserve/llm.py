"""LLM extraction: one call per prompt -> must-preserve spans + 12 whole-prompt attributes.
Uses OpenAI Structured Outputs (strict JSON schema); results are cached to JSONL."""
from __future__ import annotations

import hashlib
import json
import os

from .data import norm

SYS = """You annotate CLIENT utterances from health-behavior / motivational-interviewing (MI) counseling.
The text is an automatic speech transcript: no punctuation, fillers, repetitions and recognition errors.

1) spans: list every must-preserve piece of meaning, each copied VERBATIM from the text (short, <= 12 words),
   even if it is ungrammatical; never correct or paraphrase. One span per distinct piece of meaning.
   Do not extract style, fillers, or politeness. Cover symptoms, health conditions, emotions, life events /
   stressors, target behaviors, risks (including DENIED risk such as "I'm not suicidal"), and quantities
   (amounts, frequencies, durations, ages, probabilities).

2) Then label the WHOLE utterance (not each span):
- category: every category present: symptom, health_condition, emotion, life_event, target_behavior
  (smoking, alcohol, diet, exercise, medication, testing, sex, ...), risk, quantity.
- label: the main topic in 2-5 words (e.g. "smoking cessation", "medication adherence").
- intensity: low / moderate / high from intensifiers and wording; unspecified if no cue.
- severity: overall clinical/emotional severity of the utterance: low / moderate / high.
- risk_type: every risk type present, including denied ones (suicidal_ideation, self_harm, harm_to_others,
  substance_misuse, risky_sexual, impaired_driving, treatment_refusal, fatalism, other); ["not_risk"] if none.
- negated: true if any must-preserve meaning is stated as NOT happening / not true
  (losing the negation in translation would flip the meaning).
- experiencer: everyone whose symptom/behavior/emotion is described ("my husband still smokes" -> partner).
- temporality: current, past ("I used to bike", "I quit"), future, habitual, mixed (clearly more than one),
  or unspecified.
- hypothetical: true if the utterance contains a conditional/imagined situation ("If I sign up to be a donor...").
- certainty: certain; hedged if qualified by maybe / probably / I'm not sure / I guess / sometimes-type hedges;
  mixed if some statements are hedged and others are not.
- mi_talk: change (desire/ability/reason/need/commitment to change), sustain (arguments against change),
  ambivalent (both), neutral. mi_talk_evidence: the verbatim span that best shows it ("" if neutral).
- core_meaning: one sentence a translation must convey."""

CATEGORIES = ["symptom", "health_condition", "emotion", "life_event", "target_behavior", "risk", "quantity"]
RISK_TYPES = ["not_risk", "suicidal_ideation", "self_harm", "harm_to_others", "substance_misuse",
              "risky_sexual", "impaired_driving", "treatment_refusal", "fatalism", "other"]
EXPERIENCERS = ["client", "partner", "child", "parent", "other_family", "friend_peer", "other_person", "general"]
def enum(vals): return {"type": "string", "enum": vals}
def enum_list(vals): return {"type": "array", "items": enum(vals)}

SCHEMA = {"name": "must_preserve_prompt", "strict": True, "schema": {
  "type": "object", "additionalProperties": False,
  "required": ["spans", "category", "label", "intensity", "severity", "risk_type", "negated", "experiencer",
               "temporality", "hypothetical", "certainty", "mi_talk", "mi_talk_evidence", "core_meaning"],
  "properties": {
    "spans": {"type": "array", "items": {"type": "string"}},
    "category": enum_list(CATEGORIES),
    "label": {"type": "string"},
    "intensity": enum(["low", "moderate", "high", "unspecified"]),
    "severity": enum(["low", "moderate", "high"]),
    "risk_type": enum_list(RISK_TYPES),
    "negated": {"type": "boolean"},
    "experiencer": enum_list(EXPERIENCERS),
    "temporality": enum(["current", "past", "future", "habitual", "mixed", "unspecified"]),
    "hypothetical": {"type": "boolean"},
    "certainty": enum(["certain", "hedged", "mixed"]),
    "mi_talk": enum(["change", "sustain", "ambivalent", "neutral"]),
    "mi_talk_evidence": {"type": "string"},
    "core_meaning": {"type": "string"}}}}


ASR_LINE = "The text is an automatic speech transcript: no punctuation, fillers, repetitions and recognition errors.\n"
SYS_PAIR = SYS.replace(ASR_LINE, "")          # PAIR prompts are written text, not ASR
def sys_for(path): return SYS_PAIR if path.lower().endswith(".csv") else SYS


class LLMExtractor:
    def __init__(self, model: str = "gpt-4o-mini", cache_path: str = "outputs/llm_cache.jsonl",
                 base_url: str | None = None, retries: int = 3):
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set: copy .env.example to .env and add your key.")
        self.client = OpenAI(base_url=base_url) if base_url else OpenAI()
        self.model, self.cache_path, self.retries = model, cache_path, retries
        self.cache = self._load_cache()

    def _key(self, text: str, sys_prompt: str) -> str:  # cache invalidates if model, prompt or text change
        return hashlib.sha1(f"{self.model}\n{sys_prompt}\n{text}".encode()).hexdigest()

    def _load_cache(self) -> dict:
        out = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if "error" not in r["out"]:
                        out[r["key"]] = r
        return out

    def extract(self, text: str, sys_prompt: str = SYS) -> dict:
        text = norm(text)
        k = self._key(text, sys_prompt)
        if k in self.cache:
            return self.cache[k]
        err = None
        for _ in range(self.retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0, seed=42,
                    response_format={"type": "json_schema", "json_schema": SCHEMA},
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}])
                rec = {"key": k, "model": self.model, "out": json.loads(r.choices[0].message.content),
                       "in_tok": r.usage.prompt_tokens, "out_tok": r.usage.completion_tokens}
                os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self.cache[k] = rec
                return rec
            except Exception as e:  # noqa: BLE001
                err = e
        return {"key": k, "model": self.model, "out": {"error": str(err)}, "in_tok": 0, "out_tok": 0}
