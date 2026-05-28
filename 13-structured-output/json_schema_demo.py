"""
json_schema_demo.py — validate structured output + a repair loop.

Why this matters:
  LLMs plug into software via structured output (JSON). Two layers make it robust:
    1) a SCHEMA VALIDATOR (does the output match the required shape?)
    2) a REPAIR LOOP (when it's malformed, extract/fix or feed the error back)
  In production you'd use jsonschema/Pydantic + constrained decoding; this is a
  minimal, dependency-free version that shows the mechanism.

Run:
    python json_schema_demo.py
Stdlib only.
"""

from __future__ import annotations

import json
import re

# The schema the customer requires (subset of JSON Schema semantics).
SCHEMA = {
    "type": "object",
    "required": ["intent", "priority", "order_id"],
    "properties": {
        "intent": {"type": "string", "enum": ["refund", "status", "complaint", "other"]},
        "priority": {"type": "integer", "minimum": 1, "maximum": 5},
        "order_id": {"type": "string", "pattern": r"^ORD-\d{5}$"},
        "notes": {"type": "string"},
    },
}


def validate(obj, schema) -> list[str]:
    """Return a list of validation errors ([] == valid). Minimal validator."""
    errors: list[str] = []
    if schema["type"] == "object":
        if not isinstance(obj, dict):
            return [f"expected object, got {type(obj).__name__}"]
        for req in schema.get("required", []):
            if req not in obj:
                errors.append(f"missing required field '{req}'")
        for key, val in obj.items():
            spec = schema["properties"].get(key)
            if spec is None:
                errors.append(f"unexpected field '{key}'")
                continue
            errors.extend(_check_field(key, val, spec))
    return errors


def _check_field(key, val, spec) -> list[str]:
    errs: list[str] = []
    t = spec["type"]
    if t == "string":
        if not isinstance(val, str):
            errs.append(f"'{key}' must be string")
            return errs
        if "enum" in spec and val not in spec["enum"]:
            errs.append(f"'{key}'={val!r} not in {spec['enum']}")
        if "pattern" in spec and not re.match(spec["pattern"], val):
            errs.append(f"'{key}'={val!r} fails pattern {spec['pattern']}")
    elif t == "integer":
        if not isinstance(val, int) or isinstance(val, bool):
            errs.append(f"'{key}' must be integer")
            return errs
        if "minimum" in spec and val < spec["minimum"]:
            errs.append(f"'{key}'={val} < min {spec['minimum']}")
        if "maximum" in spec and val > spec["maximum"]:
            errs.append(f"'{key}'={val} > max {spec['maximum']}")
    return errs


def extract_json(text: str):
    """Models often wrap JSON in prose / code fences. Pull out the first object."""
    # strip code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # grab the outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None, "no JSON object found"
    try:
        return json.loads(text[start : end + 1]), None
    except json.JSONDecodeError as e:
        return None, f"json decode error: {e}"


# Simulated model responses: messy -> fixed, as a repair loop would drive it.
MODEL_ATTEMPTS = [
    'Sure! ```json\n{"intent": "refund", "priority": 9, "order_id": "ORD-12"}\n```',  # bad: priority>5, order_id pattern
    '{"intent": "refund", "priority": 2, "order_id": "ORD-00042", "notes": "wants refund"}',  # valid
]


def repair_loop(attempts: list[str], schema, max_tries: int = 3):
    """Drive validation + 'repair'. Here attempts are pre-canned; in reality each
    retry re-prompts the model WITH the validation errors so it can self-correct."""
    for i, raw in enumerate(attempts[:max_tries]):
        print(f"\n[try {i}] raw model output:\n  {raw}")
        obj, err = extract_json(raw)
        if err:
            print(f"  parse FAILED: {err}  -> re-prompt with the error")
            continue
        errors = validate(obj, schema)
        if not errors:
            print(f"  VALID -> {json.dumps(obj)}")
            return obj
        print("  schema errors (fed back to the model to fix):")
        for e in errors:
            print(f"    - {e}")
    print("\n[give up] return error to caller after max_tries (don't crash)")
    return None


def main() -> None:
    print("Schema requires: intent(enum), priority(1-5 int), order_id(ORD-#####)")
    result = repair_loop(MODEL_ATTEMPTS, SCHEMA)
    print(
        "\nProduction hardening:\n"
        "  - Prefer CONSTRAINED DECODING / JSON mode -> valid BY CONSTRUCTION\n"
        "    (vLLM outlines/xgrammar, SGLang, Fireworks JSON mode).\n"
        "  - Keep validate+repair as defense-in-depth + for SEMANTIC rules.\n"
        "  - Track parse-failure rate as a production metric (module 05).\n"
        "  - If still flaky, SFT on (input -> valid JSON) examples (module 04)."
    )
    return result  # noqa: RET504


if __name__ == "__main__":
    main()
