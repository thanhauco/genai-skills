"""
data_prep.py — build + validate an SFT chat dataset (the real bottleneck).

Why this matters:
  Fireworks' own blog: the fine-tuning bottleneck is integration friction and
  iteration speed, not the algorithm. The single most common failure is bad
  DATA: wrong format, duplicates, train/test leakage, imbal/too-few examples.

This script:
  1) Builds a small chat-format JSONL dataset (OpenAI/Fireworks SFT shape).
  2) Validates it: schema, role order, dedup, length stats, train/val leakage.

Run:
    python data_prep.py
Stdlib only. Writes sample_sft.jsonl next to this file.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sample_sft.jsonl")

# Chat-format SFT examples. Real datasets have hundreds-to-thousands of these.
RAW_EXAMPLES = [
    {
        "system": "You are a support agent for Acme Cloud. Answer in <=2 sentences and cite the doc id.",
        "user": "How do I rotate my API key?",
        "assistant": "Go to Settings > API Keys and click Rotate; the old key works for 24h. [doc:keys-101]",
    },
    {
        "system": "You are a support agent for Acme Cloud. Answer in <=2 sentences and cite the doc id.",
        "user": "What regions are available?",
        "assistant": "Acme Cloud runs in us-east, us-west, and eu-central. [doc:regions-3]",
    },
    {
        "system": "You are a support agent for Acme Cloud. Answer in <=2 sentences and cite the doc id.",
        "user": "Why am I getting a 429?",
        "assistant": "A 429 means you exceeded your rate limit; back off and retry with jitter. [doc:limits-7]",
    },
    # A deliberate duplicate (validator should catch it):
    {
        "system": "You are a support agent for Acme Cloud. Answer in <=2 sentences and cite the doc id.",
        "user": "How do I rotate my API key?",
        "assistant": "Go to Settings > API Keys and click Rotate; the old key works for 24h. [doc:keys-101]",
    },
]


def to_chat_record(ex: dict) -> dict:
    """Convert to the messages[] format SFT trainers expect."""
    return {
        "messages": [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]
    }


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _hash(messages: list[dict]) -> str:
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def validate(records: list[dict]) -> dict:
    """Return a report dict; flag the common dataset problems."""
    problems: list[str] = []
    seen: set[str] = set()
    dups = 0
    role_seq_bad = 0
    assistant_lens: list[int] = []

    for i, r in enumerate(records):
        msgs = r.get("messages")
        if not isinstance(msgs, list) or not msgs:
            problems.append(f"record {i}: missing messages[]")
            continue
        roles = [m.get("role") for m in msgs]
        # expect system?, user, assistant ending in assistant
        if roles[-1] != "assistant":
            role_seq_bad += 1
            problems.append(f"record {i}: must end with an assistant turn (got {roles})")
        if any(not m.get("content") for m in msgs):
            problems.append(f"record {i}: empty content in a turn")

        h = _hash(msgs)
        if h in seen:
            dups += 1
        seen.add(h)

        last = msgs[-1]
        if last.get("role") == "assistant":
            assistant_lens.append(len(last.get("content", "").split()))

    return {
        "n": len(records),
        "unique": len(seen),
        "duplicates": dups,
        "bad_role_sequences": role_seq_bad,
        "assistant_word_len": {
            "min": min(assistant_lens) if assistant_lens else 0,
            "max": max(assistant_lens) if assistant_lens else 0,
            "avg": round(sum(assistant_lens) / len(assistant_lens), 1) if assistant_lens else 0,
        },
        "problems": problems,
    }


def train_val_split(records: list[dict], val_frac: float = 0.25) -> tuple[list[dict], list[dict], int]:
    """Deterministic split + leakage check (same example must not be in both)."""
    k = max(1, int(len(records) * val_frac))
    val, train = records[:k], records[k:]
    train_hashes = {_hash(r["messages"]) for r in train}
    leaked = sum(1 for r in val if _hash(r["messages"]) in train_hashes)
    return train, val, leaked


def main() -> None:
    records = [to_chat_record(e) for e in RAW_EXAMPLES]
    write_jsonl(OUT, records)
    print(f"wrote {len(records)} records -> {OUT}\n")

    report = validate(records)
    print("=== validation report ===")
    print(f"records={report['n']}  unique={report['unique']}  duplicates={report['duplicates']}")
    print(f"bad_role_sequences={report['bad_role_sequences']}")
    print(f"assistant length (words): {report['assistant_word_len']}")
    if report["problems"]:
        print("problems found:")
        for p in report["problems"]:
            print("  -", p)

    train, val, leaked = train_val_split(records)
    print(f"\nsplit: train={len(train)} val={len(val)}  leakage(val-and-train)={leaked}")

    print(
        "\nField checklist before any training run:\n"
        "  [ ] format matches the trainer (messages[] roles, ends on assistant)\n"
        "  [ ] dedup exact + near-duplicate prompts\n"
        "  [ ] NO train/test leakage (your eval must be honest)\n"
        "  [ ] enough examples + balanced across the behaviors you want\n"
        "  [ ] lengths fit the model's context; outputs are the gold behavior\n"
        "  [ ] a held-out eval set that reflects PRODUCTION (module 05)"
    )


if __name__ == "__main__":
    main()
