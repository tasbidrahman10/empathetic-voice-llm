"""
qlora_setup.py — Load Qwen2.5-Omni Thinker in 4-bit NF4 and inject LoRA adapters.

Strategy:
  - The training subprocess sets CUDA_VISIBLE_DEVICES=0, so only GPU 0 is
    visible. Trainer sees 1 GPU and never uses DataParallel.
  - The full model loads onto GPU 0 in 4-bit (~5-6 GB). After extracting
    the Thinker, the rest (audio encoder etc.) is deleted to free ~1 GB.
  - Gradient checkpointing keeps activation memory to ~200 MB so the
    entire training run fits within the 14.56 GB T4 limit.
"""

import gc

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_model_for_training(config: dict):
    """
    Returns (thinker, processor) where thinker is a PEFT-wrapped CausalLM
    on cuda:0, ready for HuggingFace Trainer.
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

    # CUDA_VISIBLE_DEVICES=0 is set in the notebook subprocess env, so
    # device_map="auto" maps everything to the single visible GPU (cuda:0).
    print(f"Loading {model_id} in 4-bit NF4 on GPU 0 ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        enable_audio_output=False,
    )

    processor = Qwen2_5OmniProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Extract Thinker (standard CausalLM forward — accepts input_ids, labels).
    thinker = model.thinker
    print("Thinker extracted.")

    # Delete the rest of the model (audio encoder etc.) to reclaim ~1 GB VRAM.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    free_gb = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
    print(f"GPU 0 free after cleanup: {free_gb:.1f} GB")

    # Gradient checkpointing: recomputes activations instead of storing them.
    # Cuts activation memory from ~8 GB to ~200 MB — essential for fitting
    # training on a single T4.
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
    # Re-register after PEFT wrapping so gradient checkpointing keeps working.
    thinker.enable_input_require_grads()
    thinker.print_trainable_parameters()

    return thinker, processor
