# Capstone — a full mock Field Engineering engagement

One worked, end-to-end story that threads **every module** together the way a real AI Field Engineer engagement runs: discovery → sizing → build → measure → tune → decide → deploy → feed back. Practice narrating this in 10–15 minutes; it's the single best "show me how you think" artifact.

> **Fictional customer: "Lumen"** — an AI-native startup whose core product is an AI coding-review assistant. GenAI *is* the product. Small, fast-moving eng team. They're on AWS, self-hosting Llama-70B on vLLM, and "it's too slow and too expensive."

---

## Phase 0 — Discovery (module 10)

Run the framework; don't pitch. What I uncover:

- **Product**: a PR-review bot. Devs paste a diff; it returns structured findings (JSON) + a summary. GenAI is the core product, not a feature → **AI-native segment**.
- **Pain**: p95 end-to-end latency ~9s (devs bounce); inference bill growing faster than revenue.
- **Traffic**: ~30 RPS peak, bursty around work hours. Prompts are **large and shared** — a 3k-token system prompt + coding guidelines on **every** request; diffs add 1–4k tokens; outputs ~400 tokens of JSON.
- **Constraints**: code can't leave their AWS VPC (customer IP); p95 latency target <3s; wants cost down ~40%.
- **Success metric**: p95 E2E < 3s **and** $/review down 40% **and** review-quality (their human-rated rubric) holds.
- **Stakeholders**: VP Eng (cares about $ + reliability) and two senior engineers (in the code with me).

**Output of discovery**: a hypothesis + a scoped POC with a measurable bar.

> "Your prompts share a huge static prefix and you're on fp16 70B — I think prefix caching + fp8 + structured output gets you most of the way. Let's not guess: I'll benchmark against your traffic this week. Success = p95 E2E < 3s at ≥40% lower $/review, quality held on your rubric."

---

## Phase 1 — Baseline & sizing (modules 02, 08)

Do the math before touching anything (`02-llm-inference/kv_cache_calculator.py`):

- 70B fp16 ≈ 140GB → needs **TP across GPUs**; on 2×H100 weights alone eat ~130GB, leaving little KV cache → low concurrency → queuing → the 9s tail.
- The 3k-token system prefix is **re-prefilled every request** → wasted TTFT + cost.
- Output is JSON → constrained decoding will kill parse-retries.

Hypotheses ranked by leverage:
1. **Prefix caching** the shared 3k-token prefix (biggest TTFT + cost win).
2. **fp8** weights → frees ~65GB HBM → far more KV cache → higher concurrency (`kv_cache_calculator.py` shows ~6× concurrency jump on 2×H100).
3. **Structured output / JSON mode** → no retry loops (module 13).
4. **Speculative decoding** (code is predictable → high acceptance) → lower TPOT (module 14).

---

## Phase 2 — Baseline load test (module 09)

Replay their real traffic shape (3k shared prefix + 1–4k diff, bursty 30 RPS) with `09-load-testing/async_load_test.py`:

```
BASELINE  70B fp16, 2xH100, no prefix cache
  TTFT p95 = 2,800 ms   E2E p95 = 9,100 ms
  throughput = 1,500 tok/s   error_rate = 3% (queue timeouts)
  $/review  ≈ $0.085     -> FAILS the <3s SLO
```

This is the number the VP signs off against. Now tune **one knob at a time**.

---

## Phase 3 — Tune to the SLO (modules 02, 03, 13, 14)

| Change | Lever | Result (illustrative) |
| --- | --- | --- |
| Enable **prefix caching** | reuse 3k-prefix KV | TTFT p95 2800 → 900 ms |
| **fp8** weights | +KV cache, +concurrency, faster decode | E2E p95 → 4.2s; throughput 1,500 → 4,200 tok/s |
| Raise **max-num-seqs** to the knee | batch more without breaking p95 | throughput → 5,500 tok/s |
| **JSON mode** (constrained) | kill parse-retries | -1 retry on ~8% of calls; cleaner cost |
| **Speculative decoding** (code, low temp) | cut TPOT | E2E p95 4.2s → 2.7s |

Each change is re-measured; I keep only what holds p95 within SLO.

```
TUNED  70B fp8, 2xH100, prefix cache + JSON mode + spec decode
  TTFT p95 = 850 ms    E2E p95 = 2,700 ms   (SLO < 3s  -> PASS)
  throughput = 5,500 tok/s   error_rate = 0.2%
  $/review ≈ $0.030   (was $0.085  -> ~65% cheaper)
```

---

## Phase 4 — Eval gate (modules 05, 04)

Numbers mean nothing if quality dropped. Build the gate (`05-evaluation/`):

- **Golden set**: 200 real PRs with human-rated rubric scores (frozen, no leakage).
- **Scorers**: JSON-schema validity (deterministic) + LLM-as-judge (pairwise, position-swapped) for finding quality + groundedness.
- **Result**: fp8 + spec decoding are distribution-preserving / near-lossless → rubric score 88% → 87% (within CI). **PASS.**

> Decision: do we fine-tune? Their findings format is now reliable via constrained output, and quality held — so **no SFT needed yet** (walked the module-04 ladder, stopped early). I note a future option: **SFT-LoRA** on their accepted reviews to lift quality, gated by this same eval.

---

## Phase 5 — Build & deploy in their stack (modules 07, 08)

- Ship the config on their **EKS** GPU pool: the `07-kubernetes-infra/vllm-deployment.yaml` shape (2×H100, TP=2, fp8, prefix caching, startupProbe for slow load) + the `hpa.yaml` autoscaling on **queue depth** (not CPU), warm headroom for bursts.
- **Data stays in their VPC** (PrivateLink, no egress) — meets the IP constraint (module 08 `deployment_patterns.md`).
- I write the integration **in their codebase** as a peer to their two engineers (OpenAI-compatible client, module 03 — base_url swap, no app rewrite).
- Stand up the **load test + eval as CI gates** so a future model/quant/config change can't regress p95 or quality.

**Honest call**: I also price **Fireworks managed** at their volume. Given bursty traffic + small team + ops cost, managed is close on $/review and removes GPU oncall — I present both and let the VP choose. (This honesty is the trust the JD prizes.)

---

## Phase 6 — Feed back to product (module 10, architecture.md §5)

Signals I codify and route to Fireworks product/eng:
- "Shared-prefix coding workloads want prefix caching **on by default** + a one-flag JSON mode" → concrete product proposal.
- The 2×H100 fp8 + prefix-cache + spec-decode shape → a **reusable deployment pattern** for AI-native code-tool customers.
- A recurring rough edge in spec-decoding config → a specific, urgent product gap with a repro.

---

## The result, in one slide

```
                 BASELINE            TUNED            target
 E2E p95         9.1 s        ->     2.7 s            < 3 s     PASS
 $/review        $0.085       ->     $0.030           -40%      -65% (beat)
 quality (rubric) 88%         ->     87% (CI ok)      hold      PASS
 error rate      3%           ->     0.2%
```

Levers used: **prefix caching, fp8 quantization, continuous batching headroom, constrained JSON output, speculative decoding** — validated by **load test + eval gate**, deployed **in their VPC**, with signals fed back to **product**.

---

## How every module showed up

| Module | Where it was used |
| --- | --- |
| 01 Python | the load-test client, async fan-out, retries |
| 02 inference | sizing math, prefix caching, fp8 → KV/concurrency |
| 03 serving | vLLM knobs, OpenAI-compatible integration |
| 04 fine-tuning | walked the ladder, decided **not** to SFT (yet) |
| 05 evaluation | golden set + LLM-judge gate proved quality held |
| 06 agents | (adjacent) structured findings = function-calling discipline |
| 07 kubernetes | EKS deployment + queue-depth HPA + probes |
| 08 cloud/cost | VPC residency, $/review, managed-vs-self-host call |
| 09 load testing | the baseline + tuned numbers the VP signed off |
| 10 field | discovery, exec honesty, product feedback loop |
| 11 RAG | (would apply if they added a guidelines knowledge base) |
| 12 multimodal | (future: screenshot/diagram review) |
| 13 structured output | JSON mode killed parse-retries |
| 14 prompt/spec | prompt-cache framing + speculative decoding |

**Narrate this story and you've demonstrated the whole job.**
