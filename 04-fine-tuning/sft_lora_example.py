"""
sft_lora_example.py — LoRA/QLoRA SFT, two ways.

  PART A (always runs): a tiny, dependency-free demo of the LoRA idea so you can
          EXPLAIN it: W_eff = W + (alpha/r) * B @ A, training only A and B.
  PART B (real code, runs only if torch+peft+trl+datasets installed): the actual
          TRL SFTTrainer + PEFT LoRA setup you'd run on a GPU.

Why this matters:
  LoRA freezes the base weights and learns small low-rank adapters (~0.1-1% of
  params). Cheap, fast, swappable, multi-tenant friendly. QLoRA = 4-bit base +
  LoRA adapters, so you can fine-tune big models on modest GPUs.

Run:
    python sft_lora_example.py
"""

from __future__ import annotations


# ----------------------------- PART A: the math --------------------------------
def lora_demo() -> None:
    """Show that LoRA adds a low-rank update and trains far fewer params."""
    try:
        import numpy as np
    except Exception:
        print("(numpy not installed; skipping LoRA math demo - install numpy to see it)")
        return

    rng = np.random.default_rng(0)
    d_in, d_out, r = 1024, 1024, 8  # rank r << d
    alpha = 16

    W = rng.standard_normal((d_out, d_in)) * 0.02   # frozen base weight
    A = rng.standard_normal((r, d_in)) * 0.01        # trainable (down-proj)
    B = np.zeros((d_out, r))                          # trainable (up-proj), init 0

    scale = alpha / r
    W_eff = W + scale * (B @ A)                       # effective weight at inference

    base_params = W.size
    lora_params = A.size + B.size
    print("PART A - LoRA mechanics")
    print(f"  base weight params (frozen) : {base_params:,}")
    print(f"  LoRA params (trainable)     : {lora_params:,}  "
          f"({100*lora_params/base_params:.2f}% of base)")
    print(f"  W_eff = W + (alpha/r)*B@A    : alpha={alpha} r={r} scale={scale}")
    print(f"  (B initialized to 0 -> training starts as a no-op; stable warmup)")
    print(f"  effective weight shape       : {W_eff.shape}")


# --------------------------- PART B: the real run ------------------------------
REAL_TRAINING_CODE = r'''
# Real LoRA SFT with TRL + PEFT (run on a GPU). This is the code you'd write
# alongside a customer. Requires: pip install torch transformers peft trl datasets bitsandbytes

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import torch

model_id = "meta-llama/Llama-3.1-8B-Instruct"

# QLoRA: load the base model in 4-bit to fit big models on small GPUs
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")

# LoRA: only these adapters are trained; base stays frozen
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention proj layers
)

# Expect data in chat format (see data_prep.py -> sample_sft.jsonl)
ds = load_dataset("json", data_files="sample_sft.jsonl", split="train")

cfg = SFTConfig(
    output_dir="out-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,            # higher LR is fine for LoRA
    bf16=True,
    logging_steps=10,
    max_seq_length=2048,
    packing=True,                  # pack short examples for throughput
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft_config)
trainer.train()
trainer.save_model("out-lora")     # saves the small adapter, not the whole model

# Deploy: serve base model + load the LoRA adapter (vLLM supports --enable-lora,
# Fireworks supports uploading the adapter). Multiple adapters can share one base.
'''


def try_real_path() -> bool:
    mods = []
    for m in ("torch", "peft", "trl", "datasets"):
        try:
            __import__(m)
        except Exception:
            mods.append(m)
    if mods:
        print(f"PART B - real training is gated (missing: {', '.join(mods)}).")
        print("        Showing the code instead. Install deps + a GPU to run it.\n")
        print(REAL_TRAINING_CODE)
        return False
    print("PART B - deps present. (Not auto-running a real train here; see code below.)\n")
    print(REAL_TRAINING_CODE)
    return True


def main() -> None:
    lora_demo()
    print()
    try_real_path()
    print(
        "Interview line:\n"
        "  'LoRA trains ~0.1-1% of params as low-rank adapters on top of a frozen\n"
        "   base. QLoRA 4-bit-quantizes the base so we can tune an 8B-70B on modest\n"
        "   GPUs. Adapters are swappable, so one base serves many customer tasks.'"
    )


if __name__ == "__main__":
    main()
