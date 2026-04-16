# 10 — Interview Scenarios (Field, Discovery, System Design, Behavioral)

The other half of the role: "lead discovery conversations, align stakeholders… hold your own in executive-level conversations about architecture, strategy, and business outcomes." This module is **practice scripts** — run them out loud.

## Files

| File | What it's for |
| --- | --- |
| [`discovery_call_framework.md`](discovery_call_framework.md) | How to run a sharp discovery call; question bank; qualification |
| [`system_design_scenarios.md`](system_design_scenarios.md) | 5 LLM-infra design prompts with model answers |
| [`behavioral_prep.md`](behavioral_prep.md) | STAR stories mapped to the JD; the FDE-specific questions |
| [`fireworks_specific.md`](fireworks_specific.md) | Company facts, their research, smart questions to ask them |
| [`mock_interview_questions.md`](mock_interview_questions.md) | 60+ questions across all modules to self-quiz |

## How to practice

1. **Discovery role-play**: have someone play a customer; run the framework; they grade whether you uncovered the real pain before pitching.
2. **System design out loud**: pick a scenario, whiteboard it in 20–30 min using the `architecture.md` diagrams, end at **$/1M tokens + a feedback loop**.
3. **Behavioral**: write each STAR story once, then deliver it in <2 minutes.
4. **Rapid-fire**: shuffle `mock_interview_questions.md`; 2 minutes per answer.

## The role's dual mandate (frame every answer around this)

```
   HANDS-ON-KEYBOARD                         EXECUTIVE-CREDIBLE
   build POCs/MVPs in their repo             discovery, stakeholder alignment
   benchmark, debug, deploy, tune            architecture + business outcomes
   fine-tune + eval pipelines                translate pain -> product roadmap
            \                                        /
             \____ "the quality of your engineering is the relationship" ____/
```

Great answers show **both**: you can debug a latency issue with their ML engineer *and* explain the $/1M-token trade-off to their VP that afternoon.

## The one-line summary of how you operate

> "I start at the customer's **traffic profile and success criteria**, build the smallest thing that proves value in **their** codebase, **measure** it (eval + load test), tune to the **SLO and $/1M**, and feed what I learned back into the **product**. Credibility comes from what I ship alongside them."

## Quick self-check before any interview

- [ ] I can whiteboard the serving stack + the KV-cache math ([architecture.md](../architecture.md)).
- [ ] I can run a discovery call without pitching in the first 10 minutes.
- [ ] I can pick SFT vs DPO vs RFT for a scenario and justify it.
- [ ] I can produce a latency/throughput/$ baseline and read the tail.
- [ ] I can debug a Pending/CrashLoop GPU pod from memory.
- [ ] I have 6 STAR stories that hit: shipped-in-prod, ambiguity, exec comms, conflict, failure, product-feedback.
- [ ] I have 5 sharp questions to ask **them**.
