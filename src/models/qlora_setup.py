"""
qlora_setup.py — Load Qwen2.5-Omni Thinker in 4-bit NF4 and inject LoRA adapters.

Only the Thinker (LM backbone) is trained. Audio encoder, Talker, and Token2Wav
parameters are frozen before LoRA injection.
"""

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_model_for_training(config: dict):
    """
    Load Qwen2.5-Omni in 4-bit NF4, freeze non-Thinker components, inject LoRA.

    Args:
        config: Parsed configs/config.yaml as a dict.

    Returns:
        (model, processor) — PEFT-wrapped model ready for Trainer, and the
        Qwen2_5OmniProcessor used to tokenize inputs.
    """
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    model_id = config["model"]["model_id"]
    qlora_cfg = config["qlora"]
    lora_cfg = config["lora"]

    compute_dtype = torch.bfloat16 if qlora_cfg["bnb_4bit_compute_dtype"] == "bfloat16" else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg["load_in_4bit"],
        bnb_4bit_quant_type=qlora_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora_cfg["bnb_4bit_use_double_quant"],
    )

    print(f"Loading {model_id} in 4-bit NF4 ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        # Talker has a config bug (missing pad_token_id) on some transformers versions.
        # We don't need it for text-only Thinker training.
        enable_audio_output=False,
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Extract Thinker — it's a standard CausalLM with forward(input_ids, labels, ...).
    # The full Qwen2_5OmniForConditionalGeneration forward() expects multimodal inputs
    # and doesn't accept plain input_ids, so HuggingFace Trainer can't use it directly.
    thinker = model.thinker
    print("Thinker extracted. Applying QLoRA ...")

    # Required by PEFT before LoRA injection on a 4-bit quantised model.
    thinker = prepare_model_for_kbit_training(thinker, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        # task_type intentionally omitted — TaskType enum raises NotImplementedError
        # for multimodal models (same pattern used in Phase 1 verification).
    )

    thinker = get_peft_model(thinker, lora_config)
    thinker.print_trainable_parameters()

    return thinker, processor
