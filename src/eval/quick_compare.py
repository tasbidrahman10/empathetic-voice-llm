#!/usr/bin/env python3
"""Quick base-vs-fine-tuned response comparison for EFSM.

The script loads Qwen2.5-7B-Instruct in 4-bit, generates base responses for a
fixed prompt set, unloads it, then reloads the same base model with the trained
LoRA adapter and generates fine-tuned responses. This keeps memory low enough
for a single T4 and produces a CSV table for the project demo/report.
"""

import argparse
import csv
import gc
import os
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_PROMPTS = [
    {
        "id": "sadness_job",
        "emotion": "sad",
        "prompt": "I just found out I did not get the internship I worked so hard for. I feel like I am not good enough.",
    },
    {
        "id": "anxiety_exam",
        "emotion": "anxious",
        "prompt": "My final exam is tomorrow and my chest feels tight. I keep thinking I am going to fail.",
    },
    {
        "id": "loneliness_uni",
        "emotion": "lonely",
        "prompt": "Everyone at university seems to have their own group. I feel invisible most days.",
    },
    {
        "id": "guilt_family",
        "emotion": "guilty",
        "prompt": "I snapped at my mother today even though she was only trying to help me. I feel awful about it.",
    },
    {
        "id": "anger_friend",
        "emotion": "angry",
        "prompt": "My friend shared something private about me. I am so angry and I do not know how to face them.",
    },
    {
        "id": "shame_mistake",
        "emotion": "ashamed",
        "prompt": "I made a mistake during my presentation and now I cannot stop replaying it in my head.",
    },
    {
        "id": "grief_loss",
        "emotion": "devastated",
        "prompt": "Someone close to me passed away recently. I feel numb and lost.",
    },
    {
        "id": "burnout_work",
        "emotion": "overwhelmed",
        "prompt": "I have assignments, family pressure, and project deadlines all at once. I feel like I cannot breathe.",
    },
    {
        "id": "fear_future",
        "emotion": "afraid",
        "prompt": "I am scared that I will disappoint everyone if I cannot build a good career.",
    },
    {
        "id": "rejection_relationship",
        "emotion": "rejected",
        "prompt": "Someone I really liked told me they do not feel the same way. I feel embarrassed and unwanted.",
    },
    {
        "id": "self_doubt",
        "emotion": "insecure",
        "prompt": "When I compare myself with my classmates, I feel like I am always behind.",
    },
    {
        "id": "joy_success",
        "emotion": "joyful",
        "prompt": "I finally finished a difficult milestone in my project, and I feel proud but also exhausted.",
    },
]


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(model_id: str, adapter_repo: str | None, hf_token: str | None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        token=hf_token,
    )
    if adapter_repo:
        model = PeftModel.from_pretrained(model, adapter_repo, token=hf_token)

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, system_prompt: str, emotion: str, prompt: str, max_new_tokens: int) -> str:
    user_content = f"[emotion: {emotion}] {prompt}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="results/quick_eval_results.csv")
    parser.add_argument("--limit", type=int, default=len(DEFAULT_PROMPTS))
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_id = cfg["model"]["model_id"]
    adapter_repo = cfg["model"]["hf_hub_checkpoint_repo"]
    system_prompt = cfg["system_prompt"].strip()
    hf_token = os.environ.get("HF_TOKEN")
    prompts = DEFAULT_PROMPTS[: args.limit]

    print(f"Model: {model_id}")
    print(f"Adapter: {adapter_repo}")
    print(f"Prompts: {len(prompts)}")

    print("\nLoading base model...")
    base_model, tokenizer = load_model_and_tokenizer(model_id, adapter_repo=None, hf_token=hf_token)
    rows = []
    for item in prompts:
        print(f"Base: {item['id']}")
        rows.append(
            {
                "id": item["id"],
                "emotion": item["emotion"],
                "prompt": item["prompt"],
                "base_response": generate_response(
                    base_model,
                    tokenizer,
                    system_prompt,
                    item["emotion"],
                    item["prompt"],
                    args.max_new_tokens,
                ),
            }
        )
    release_model(base_model)

    print("\nLoading fine-tuned model...")
    tuned_model, tokenizer = load_model_and_tokenizer(model_id, adapter_repo=adapter_repo, hf_token=hf_token)
    for row in rows:
        print(f"Fine-tuned: {row['id']}")
        row["fine_tuned_response"] = generate_response(
            tuned_model,
            tokenizer,
            system_prompt,
            row["emotion"],
            row["prompt"],
            args.max_new_tokens,
        )
    release_model(tuned_model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "emotion", "prompt", "base_response", "fine_tuned_response"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved comparison table to {output_path}")


if __name__ == "__main__":
    main()
