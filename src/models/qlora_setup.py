"""
qlora_setup.py — Load Qwen2.5-7B-Instruct in 4-bit NF4 and inject LoRA adapters.

We load Qwen2.5-7B-Instruct directly (not via Qwen2.5-Omni) because:
  - Qwen2.5-7B is architecturally identical to the Thinker inside Qwen2.5-Omni.
  - Loading the full Omni model (9-10B with audio encoder + Talker) exhausts
    the T4's 14.56 GB even in 4-bit due to loading overhead.
  - For text-based empathy fine-tuning, the base LLM weights are what matter.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_model_for_training(config: dict):
    """Returns (model, tokenizer) where model is PEFT-wrapped and ready for Trainer."""
    model_id = config["model"]["model_id"]
    qlora_cfg = config["qlora"]
    lora_cfg = config["lora"]

    compute_dtype = (
        torch.float16
        if qlora_cfg["bnb_4bit_compute_dtype"] == "float16"
        else torch.bfloat16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg["load_in_4bit"],
        bnb_4bit_quant_type=qlora_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora_cfg["bnb_4bit_use_double_quant"],
    )

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU 0 total VRAM: {total_gb:.1f} GB  |  compute_dtype: {compute_dtype}")
    print(f"Loading {model_id} in 4-bit NF4 ...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    free_gb = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
    print(f"GPU 0 free after model load: {free_gb:.1f} GB")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    return model, tokenizer
