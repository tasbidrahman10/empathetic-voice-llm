"""
train.py — QLoRA fine-tuning of Qwen2.5-Omni Thinker on EmpatheticDialogues.

Usage:
    python src/training/train.py --config configs/config.yaml
"""

import os

# Must be set before ANY torch/CUDA import — restricts CUDA to GPU 0 only so
# Trainer sees device_count()=1 and never wraps the model in DataParallel.
# BnB 4-bit models break under DataParallel (weights stay on cuda:0 while
# DataParallel copies inputs to cuda:1).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import sys

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # W&B project name must be set before importing wandb/Trainer.
    wandb_cfg = config.get("wandb", {})
    os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "efsm-cse465")
    if wandb_cfg.get("entity"):
        os.environ["WANDB_ENTITY"] = wandb_cfg["entity"]

    # ── Model + processor ────────────────────────────────────────────────────
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.models.qlora_setup import load_model_for_training
    from src.data.dataset import EFSMDataset

    model, tokenizer = load_model_for_training(config)

    # ── Datasets ─────────────────────────────────────────────────────────────
    data_cfg = config["data"]
    max_seq_len = data_cfg["max_seq_len"]

    train_dataset = EFSMDataset(data_cfg["train_path"], tokenizer, max_seq_len)
    eval_dataset = EFSMDataset(data_cfg["val_path"], tokenizer, max_seq_len)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(eval_dataset)}")

    # ── Training arguments ───────────────────────────────────────────────────
    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

    train_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        warmup_steps=train_cfg["warmup_steps"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        max_grad_norm=train_cfg["max_grad_norm"],
        report_to=train_cfg["report_to"],
        run_name=train_cfg["run_name"],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # ── Callback: upload each checkpoint to HF Hub so Colab disconnects don't lose progress ──
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from transformers import TrainerCallback
        from huggingface_hub import HfApi

        repo_id = config["model"]["hf_hub_checkpoint_repo"]
        _api = HfApi()

        class HubCheckpointCallback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                if os.path.isdir(ckpt):
                    _api.upload_folder(
                        folder_path=ckpt,
                        repo_id=repo_id,
                        path_in_repo=f"checkpoint-{state.global_step}",
                        repo_type="model",
                        token=hf_token,
                        commit_message=f"checkpoint-{state.global_step}",
                    )
                    print(f"Checkpoint {state.global_step} uploaded to HF Hub.")

        trainer.add_callback(HubCheckpointCallback())

    # ── Train ────────────────────────────────────────────────────────────────
    print("Starting QLoRA fine-tuning ...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── Save LoRA adapter ────────────────────────────────────────────────────
    adapter_dir = os.path.join(train_cfg["output_dir"], "final-adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to {adapter_dir}")

    # ── Upload to HuggingFace Hub ─────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN not set — skipping HuggingFace Hub upload.")
        return

    from huggingface_hub import HfApi

    repo_id = config["model"]["hf_hub_checkpoint_repo"]
    api = HfApi()
    api.upload_folder(
        folder_path=adapter_dir,
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
        commit_message="Phase 3: final LoRA adapter after training",
    )
    print(f"Adapter uploaded to HuggingFace Hub: {repo_id}")


if __name__ == "__main__":
    main()
