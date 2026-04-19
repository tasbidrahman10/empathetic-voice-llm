"""
qlora_setup.py — Load Qwen2.5-Omni Thinker in 4-bit NF4 and inject LoRA adapters.

Strategy:
  1. Load the full model across both T4 GPUs (device_map="auto") so it fits in VRAM.
  2. Extract model.thinker, strip accelerate's device hooks, move to GPU 0 alone.
  3. Delete the full model (frees audio encoder + other components from both GPUs).
  4. Apply QLoRA to the Thinker on GPU 0 — no cross-device issues, no DataParallel.
"""

import gc

import torch
from accelerate.hooks import remove_hook_from_submodules
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_model_for_training(config: dict):
    """
    Returns (thinker, processor) where thinker is a PEFT-wrapped CausalLM
    sitting entirely on cuda:0, ready for HuggingFace Trainer.
    """
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    model_id = config["model"]["model_id"]
    qlora_cfg = config["qlora"]
    lora_cfg = config["lora"]

    compute_dtype = (
        torch.bfloat16
        if qlora_cfg["bnb_4bit_compute_dtype"] == "bfloat16"
        else torch.float16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg["load_in_4bit"],
        bnb_4bit_quant_type=qlora_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora_cfg["bnb_4bit_use_double_quant"],
    )

    # Step 1 — Load full model across both GPUs so it fits in VRAM.
    print(f"Loading {model_id} in 4-bit NF4 across both GPUs ...")
    full_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        enable_audio_output=False,
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Step 2 — Extract Thinker and remove accelerate's per-layer device hooks.
    # These hooks are added by device_map="auto" to move tensors between GPUs.
    # PEFT replaces the hooked Linear modules, breaking the hooks and causing
    # cross-device bmm errors. We remove hooks first, then consolidate to GPU 0.
    print("Extracting Thinker and consolidating to GPU 0 ...")
    thinker = full_model.thinker
    remove_hook_from_submodules(thinker)
    thinker = thinker.to("cuda:0")

    # Step 3 — Delete the rest of the model (audio encoder etc.) to free VRAM.
    # Thinker in 4-bit is ~3.5 GB; GPU 0 has 14.56 GB — plenty of room.
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU 0 free after cleanup: "
          f"{(torch.cuda.mem_get_info(0)[0] / 1024**3):.1f} GB")

    # Step 4 — Apply QLoRA to the single-GPU Thinker.
    thinker = prepare_model_for_kbit_training(thinker, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        # task_type omitted — avoids NotImplementedError on multimodal model classes.
    )

    thinker = get_peft_model(thinker, lora_config)
    thinker.enable_input_require_grads()
    thinker.print_trainable_parameters()

    return thinker, processor
