# Hyperscaler AI platforms — SageMaker / Bedrock, Azure AI Foundry, Vertex

The JD lists these as preferred. You should know what each is, when a customer uses it, and how Fireworks fits alongside them. Fireworks **partnered with Azure AI Foundry**, so know that one well.

## AWS

### Amazon Bedrock
- **What**: serverless, fully-managed access to foundation models (Anthropic, Meta, Mistral, Amazon Titan, etc.) via a single API. No infrastructure.
- **Use when**: team wants managed models, pay-per-token, fast integration, AWS-native (IAM, PrivateLink, guardrails). Also: Knowledge Bases (RAG), Agents, fine-tuning of supported models.
- **Trade-off**: less control over serving internals; model menu is curated.

### Amazon SageMaker
- **What**: end-to-end ML platform — train, tune, and **deploy your own models** on managed GPU **endpoints** (real-time, async, batch). SageMaker JumpStart has prebuilt model deployments.
- **Use when**: customer brings their own / open-weight model, needs training + custom serving on managed GPU infra without running raw EC2.
- **Trade-off**: more knobs than Bedrock; you manage the model + container (DJL, TGI, vLLM containers supported).

## Azure

### Azure AI Foundry (formerly Azure AI Studio)
- **What**: Azure's unified platform to discover (model catalog), deploy (managed/serverless endpoints), evaluate, and build **agents** with models — incl. Azure OpenAI and a broad open model catalog.
- **Use when**: Azure-native customers; enterprise governance, content safety, identity (Entra), private networking; agent + RAG tooling.
- **Fireworks angle**: **Fireworks partnered with Azure AI Foundry** — customers can access Fireworks' fast inference + fine-tuning within the Azure ecosystem. Great talking point: meet enterprises where they already are (Azure) while delivering Fireworks performance.

## GCP

### Vertex AI
- **What**: Google Cloud's managed ML platform — **Model Garden** (Gemini + open models like Llama/Gemma), managed training, and **prediction endpoints** for serving on managed GPUs/TPUs.
- **Use when**: GCP-native customers; want Gemini or open models with managed training/serving, MLOps (pipelines, model registry, evaluation), and TPU options.

## Side-by-side

| | Bedrock | SageMaker | Azure AI Foundry | Vertex AI |
| --- | --- | --- | --- | --- |
| Primary mode | Serverless model API | Deploy your own on managed endpoints | Catalog + managed/serverless deploy + agents | Model Garden + managed endpoints |
| Bring-your-own model | Limited (curated + some custom) | Yes | Yes (catalog + custom) | Yes |
| Fine-tuning | Supported models | Full control | Supported | Supported |
| Best for | Fast managed API on AWS | Custom serving/training on AWS | Enterprise Azure + agents | GCP + Gemini/TPU |
| Infra ops | None | Some | Low | Low–some |

## Where Fireworks fits (the pitch)

- **Performance**: independently benchmarked fastest LLM inference; FireAttention + tuned serving beat naive deployments.
- **Breadth**: serving + fine-tuning (SFT/DPO/RFT) + function-calling + multimodal in one platform.
- **Simplicity**: OpenAI-compatible API; managed scaling; no GPU ops.
- **Ecosystem reach**: available via partners (e.g., **Azure AI Foundry**) so enterprises consume it where they already operate.
- **Honest positioning vs hyperscalers**: hyperscalers are great defaults; Fireworks competes on **speed, cost-efficiency, and fine-tuning/iteration velocity**, and can run inside or alongside the customer's cloud.

## FDE framing in a customer conversation

> "If you're standardized on Azure, you can consume Fireworks through Azure AI Foundry — same governance and networking, but our inference speed and fine-tuning loop. If you're on AWS and self-hosting on SageMaker/EKS, we'll benchmark your config against Fireworks managed on a $/1M-token basis at your SLO and let the numbers decide."
