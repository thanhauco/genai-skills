# Discovery call framework

The JD: "Lead structured discovery conversations to unpack customer pain points, constraints, and success criteria before proposing solutions." A great Field Engineer **earns the right to prescribe** by diagnosing first. Don't pitch in the first 10 minutes.

## The arc of a good discovery call

```
 1. Frame (2 min)      who you are, goal of the call, "mind if I ask a lot of questions?"
 2. Context (5 min)    their product, where GenAI sits, why now
 3. Pain (10 min)      the actual problem; dig past the first answer
 4. Constraints (5)    latency SLO, budget, data rules, stack, timeline, team
 5. Success (5 min)    what "this worked" looks like, measurably
 6. Quantify (3 min)   cost of the problem / value of solving it
 7. Next step (2 min)  concrete: a scoped POC with a success metric + a date
```

You're listening for: the **real** bottleneck, who decides, what "good" means in numbers, and whether this is a fast-moving AI-native team (the segment) where you'll be in the code.

## Question bank (steal these)

**Context**
- "Walk me through what your product does and where the model sits in the experience."
- "Is GenAI the core product or a feature?" (AI-native segment → former)
- "What changed that made this a priority now?"

**Pain (the 5-whys zone)**
- "Where does it hurt today — latency, quality, cost, scale, or iteration speed?"
- "Show me the part of the pipeline that's slow/expensive/unreliable."
- "What have you already tried? What happened?"
- "If nothing changes, what breaks in 3 months?"

**Constraints**
- "What's your latency SLO — p95 TTFT and end-to-end?"
- "What traffic are you serving — RPS, prompt/output lengths, peak vs average?"
- "What's the cost envelope? Are you cost- or latency-bound right now?"
- "Where can data live? Any compliance (HIPAA/GDPR/SOC2) or VPC constraints?"
- "What's your current stack — model, serving, cloud, orchestration?"
- "Who's on the team and what's the timeline?"

**Success criteria**
- "How will we know this worked? What metric moves?"
- "What does production-ready mean to you?"
- "Who signs off, and what do they care about?"

**Quantify**
- "What's this problem costing you — in dollars, churn, or engineering time?"
- "What's it worth to cut latency in half / cost by 40% / ship 2 weeks sooner?"

## Frameworks to structure it (name-drop-able)

- **MEDDIC / MEDDPICC** — Metrics, Economic buyer, Decision criteria, Decision process, Identify pain, Champion. Great for qualifying enterprise deals.
- **SPIN** — Situation, Problem, Implication, Need-payoff. Great for the questioning flow above.
- **JTBD (Jobs To Be Done)** — "what job is the customer hiring the model to do?"

For the **AI-native segment**, deals move fast with fewer stakeholders — keep it lightweight, get to the code, but still pin down the **success metric** and **economic value**.

## Anti-patterns (what loses trust)

- **Pitching before diagnosing.** You sound like sales, not an engineer.
- **Happy-ears.** Hearing "it's a latency problem" and skipping the why (often it's the prompt/architecture, not the GPU).
- **No metric.** Leaving without a measurable success criterion = no way to prove value.
- **Solutioning the symptom.** They ask for a bigger GPU; the fix is prefix caching.
- **No next step.** A great call ends with a scoped POC, an owner, and a date.

## The transition from discovery to building

> "Here's what I heard: your p95 TTFT is 1.2s on 8k-token RAG prompts, it's costing you conversions, and you're on vLLM/A100s. I think the win is prefix-caching the shared context plus fp8 — but let's not guess. I'll stand up a benchmark against your traffic this week; success = p95 TTFT under 400ms at equal or lower cost, quality held on your eval. Sound right?"

That sentence does discovery → hypothesis → measurable POC → mutual agreement. That's the job.

## Reading: map to other modules

- Constraints/SLO → modules 02, 09 (inference math, load test).
- Model/fine-tuning strategy → modules 04, 05.
- Stack/deploy → modules 03, 07, 08.
- Turning pain into product signal → the feedback loop in [architecture.md](../architecture.md) §5.
