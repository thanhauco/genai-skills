# Mock interview question bank (60+)

Self-quiz across every skill. Shuffle and answer out loud, ~2 minutes each. Answers/pointers are in the linked modules.

## Python & engineering (module 01)
1. `asyncio` vs threads vs processes — when each? (GIL, I/O vs CPU)
2. How do you bound concurrency against a rate-limited endpoint?
3. Implement retries with exponential backoff + jitter. Why jitter?
4. What's a circuit breaker and when do you add one?
5. A script is "slow" — your first three diagnostic moves?
6. How do you measure TTFT from a streaming response?
7. cProfile vs timeit vs tracemalloc — what does each tell you?
8. Make a generator that batches an iterable into chunks of n.
9. Implement an LRU cache. Where would you use one in an LLM app?
10. How do you structure logs so they're useful in production?

## LLM inference (module 02)
11. Why is prefill compute-bound but decode memory-bandwidth-bound?
12. Write the KV-cache-bytes-per-token formula. What shrinks it? (GQA/MQA, quant)
13. Estimate max concurrency for a 13B fp16 model on an 80GB H100.
14. What is continuous batching and why does it beat static batching?
15. Total latency in terms of TTFT and TPOT?
16. How does quantization improve throughput (two mechanisms)?
17. fp16 vs fp8 vs int4 — quality/memory/speed trade-offs?
18. What's the decode TPOT floor and how do you estimate it?
19. How does context length affect memory and latency?
20. Speculative decoding — what does it help (TTFT or TPOT)?
21. Tensor vs pipeline parallelism — when each?
22. What is prefix caching / RadixAttention and which workloads love it?

## Serving frameworks (module 03)
23. vLLM vs SGLang vs TensorRT-LLM — pick one for a shared-prefix agent workload.
24. First knobs you tune on vLLM?
25. When is TensorRT-LLM's complexity worth it?
26. Why does OpenAI-compatibility matter to a customer?
27. How do you validate a new model family on a framework?
28. What does `--max-num-seqs` trade off?

## Fine-tuning (module 04)
29. Walk the ladder: prompt → … → RFT. When do you stop?
30. SFT vs DPO vs RFT — give a scenario for each.
31. LoRA vs full fine-tune — trade-offs? What's QLoRA?
32. Why is "the bottleneck not the algorithm"? What is it?
33. Write the DPO loss in words. What does beta control?
34. When is RFT worth it? What's a verifiable reward?
35. A customer's fine-tune "isn't working" — your first questions?
36. How do you prevent train/eval leakage and why does it matter?

## Evaluation (module 05)
37. Why aren't benchmark scores enough?
38. Deterministic scoring vs LLM-as-judge — when each?
39. Name 3 LLM-as-judge biases and a mitigation for each.
40. How do you build an eval that reflects production?
41. What's pass@k and when do you use it?
42. Why report a confidence interval on accuracy?
43. How do you stop quality regressions after handoff?
44. How do you eval a RAG system vs an agent?

## Agents (module 06)
45. Describe the function-calling loop end to end.
46. An agent loops forever — how do you make it robust?
47. How do you make tool calls reliable?
48. How do you evaluate an agent (trajectory, not answer)?
49. How do you cut agent cost/latency?
50. Security risks of tool use and your mitigations (incl. prompt injection)?

## Kubernetes / GPU (module 07)
51. A GPU pod is Pending — diagnose.
52. CrashLoopBackOff right after start — likely causes + fixes?
53. Why is CPU% a bad autoscale trigger for LLM serving? What's better?
54. What's special about `nvidia.com/gpu` requests/limits?
55. Why a startupProbe for model servers?
56. How do you run one big model across 4 GPUs in K8s?

## Cloud / cost (module 08)
57. Convert a GPU hourly rate to $/1M tokens.
58. Managed (Bedrock/Foundry/Vertex/Fireworks) vs self-host — decide.
59. Pick a GPU + instance for Llama-70B low-latency on AWS.
60. How do data-residency rules change the design?
61. When do you use spot/preemptible GPUs?
62. What is Azure AI Foundry and why does the Fireworks partnership matter?

## Load testing (module 09)
63. How do you establish a latency/throughput/cost baseline?
64. Throughput is great but p99 is awful — what's happening?
65. How do you find the saturation point?
66. What makes a load test misleading?
67. Why report the tail (p95/p99) not the mean? What's goodput?

## Field / behavioral (module 10)
68. Run a discovery call — what's your opening and first 3 questions?
69. Tell me about shipping code in a customer's production environment.
70. A VP asks why their inference bill is high — walk me through it.
71. Describe turning a field signal into a product change.
72. Disagreeing with a customer's technical direction — example?
73. Why field engineering over pure SWE or pure sales?
74. Why Fireworks? Connect one of their blogs to something you'd do.

## Scoring yourself
- **Confident + specific + quantified** → green.
- **Right idea, fuzzy on numbers** → re-read the module, redo the math by hand.
- **Blanked** → that module is your next study block.

Target: green on every Python + inference + one field question before interviewing.
