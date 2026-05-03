#!/usr/bin/env python3
"""Quick base-vs-fine-tuned response comparison for EFSM.

The preferred path loads one Qwen2.5-7B-Instruct model with the trained LoRA
adapter, then temporarily disables the adapter to produce the base response.
This gives a fair comparison without loading two separate 7B models.
"""

import argparse
import csv
import gc
import os
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


THERAPEUTIC_SYSTEM_PROMPT = """You are an empathetic therapeutic conversation partner.
Respond like a warm, careful counselor in 3 to 5 sentences.
First acknowledge the user's emotion clearly.
Then validate why that feeling makes sense.
Then gently invite them to share more or reflect, without rushing into advice.
Do not give generic reassurance such as "you'll be fine."
Do not immediately problem-solve unless the user asks for solutions.
Make the person feel heard, understood, and emotionally safe."""


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_qwen_device_map(model_id: str, hf_token: str | None, strategy: str):
    """Build a memory-safe device map for quick evaluation.

    Kaggle sometimes exposes two T4s but the process still packs generation onto
    cuda:0. The safest demo path is CPU offload: keep early blocks on cuda:0 and
    move later blocks + output head to CPU. It is slower, but it avoids the
    single-T4 memory cliff.
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    num_layers = int(config.num_hidden_layers)

    if strategy == "single_gpu_4bit":
        max_memory = {0: "13GiB", "cpu": "48GiB"}
        return {"": 0}, max_memory

    if strategy == "auto":
        max_memory = {i: "12GiB" for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "48GiB"
        return "auto", max_memory

    if strategy == "two_gpu" and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        split = num_layers // 2
        device_map = {
            "model.embed_tokens": 0,
            "model.norm": 1,
            "lm_head": 1,
        }
        for layer_idx in range(num_layers):
            device_map[f"model.layers.{layer_idx}"] = 0 if layer_idx < split else 1

        max_memory = {0: "13GiB", 1: "13GiB", "cpu": "48GiB"}
        print(f"Using explicit two-GPU map: layers 0-{split - 1} on cuda:0, {split}-{num_layers - 1} on cuda:1")
        return device_map, max_memory

    gpu_layers = max(8, num_layers // 3)

    device_map = {
        "model.embed_tokens": 0,
        "model.norm": "cpu",
        "lm_head": "cpu",
    }
    for layer_idx in range(num_layers):
        device_map[f"model.layers.{layer_idx}"] = 0 if layer_idx < gpu_layers else "cpu"

    max_memory = {0: "10GiB", "cpu": "96GiB"}
    print(f"Using CPU-offload map: layers 0-{gpu_layers - 1} on cuda:0, {gpu_layers}-{num_layers - 1} on CPU")
    return device_map, max_memory


def load_model_and_tokenizer(model_id: str, adapter_repo: str | None, hf_token: str | None, strategy: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Visible CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    device_map, max_memory = build_qwen_device_map(model_id, hf_token, strategy)

    load_kwargs = {
        "device_map": device_map,
        "max_memory": max_memory,
        "trust_remote_code": True,
        "token": hf_token,
        "low_cpu_mem_usage": True,
        "offload_folder": "offload",
    }
    if strategy == "cpu_offload":
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **load_kwargs,
    )
    if adapter_repo:
        model = PeftModel.from_pretrained(model, adapter_repo, token=hf_token)

    model.eval()
    model.config.use_cache = False
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
    first_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(first_device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            use_cache=False,
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


def run_single_peft_comparison(
    model_id: str,
    adapter_repo: str,
    system_prompt: str,
    hf_token: str | None,
    prompts: list[dict],
    max_new_tokens: int,
    device_strategy: str,
) -> list[dict]:
    """Compare base vs tuned responses from one loaded PEFT model."""
    print("\nLoading one base model with LoRA adapter...")
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        adapter_repo=adapter_repo,
        hf_token=hf_token,
        strategy=device_strategy,
    )

    rows = []
    for item in prompts:
        print(f"Compare: {item['id']}")
        with model.disable_adapter():
            base_response = generate_response(
                model,
                tokenizer,
                system_prompt,
                item["emotion"],
                item["prompt"],
                max_new_tokens,
            )

        fine_tuned_response = generate_response(
            model,
            tokenizer,
            system_prompt,
            item["emotion"],
            item["prompt"],
            max_new_tokens,
        )

        rows.append(
            {
                "id": item["id"],
                "emotion": item["emotion"],
                "prompt": item["prompt"],
                "base_response": base_response,
                "fine_tuned_response": fine_tuned_response,
            }
        )

    release_model(model)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="results/quick_eval_results.csv")
    parser.add_argument("--limit", type=int, default=len(DEFAULT_PROMPTS))
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--device-strategy",
        choices=["single_gpu_4bit", "cpu_offload", "two_gpu", "auto"],
        default="single_gpu_4bit",
        help="single_gpu_4bit is best for Colab/Kaggle when using --compare-mode single_peft.",
    )
    parser.add_argument(
        "--compare-mode",
        choices=["single_peft", "sequential"],
        default="single_peft",
        help="single_peft compares base vs tuned with one loaded PEFT model.",
    )
    parser.add_argument(
        "--system-prompt-style",
        choices=["config", "therapeutic"],
        default="config",
        help="Use config prompt or a stronger therapeutic evaluation prompt.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_id = cfg["model"]["model_id"]
    adapter_repo = cfg["model"]["hf_hub_checkpoint_repo"]
    system_prompt = (
        THERAPEUTIC_SYSTEM_PROMPT
        if args.system_prompt_style == "therapeutic"
        else cfg["system_prompt"].strip()
    )
    hf_token = os.environ.get("HF_TOKEN")
    prompts = DEFAULT_PROMPTS[: args.limit]

    print(f"Model: {model_id}")
    print(f"Adapter: {adapter_repo}")
    print(f"Prompts: {len(prompts)}")
    print(f"System prompt style: {args.system_prompt_style}")

    if args.compare_mode == "single_peft":
        rows = run_single_peft_comparison(
            model_id=model_id,
            adapter_repo=adapter_repo,
            system_prompt=system_prompt,
            hf_token=hf_token,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            device_strategy=args.device_strategy,
        )
    else:
        print("\nLoading base model...")
        base_model, tokenizer = load_model_and_tokenizer(
            model_id,
            adapter_repo=None,
            hf_token=hf_token,
            strategy=args.device_strategy,
        )
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
        tuned_model, tokenizer = load_model_and_tokenizer(
            model_id,
            adapter_repo=adapter_repo,
            hf_token=hf_token,
            strategy=args.device_strategy,
        )
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
