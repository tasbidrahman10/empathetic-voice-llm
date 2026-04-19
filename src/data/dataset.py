"""
EFSMDataset — PyTorch Dataset for QLoRA fine-tuning of Qwen2.5-Omni Thinker.

Loads pre-processed JSONL files (each line = a JSON array of ChatML message dicts),
tokenizes with the Qwen processor, and applies label masking so cross-entropy loss
is computed only on assistant-turn tokens.
"""

import json

import torch
from torch.utils.data import Dataset


class EFSMDataset(Dataset):
    """
    Args:
        jsonl_path: Path to a JSONL file where each line is a JSON array of
                    message dicts [{role, content}, ...].
        tokenizer:  Qwen2_5OmniProcessor (or any tokenizer with
                    apply_chat_template).
        max_seq_len: Sequences longer than this are truncated from the right.
    """

    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data: list[list[dict]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        messages = self.data[idx]

        # Use apply_chat_template(tokenize=False) + encode() instead of
        # apply_chat_template(tokenize=True) because fast tokenizers can return
        # a tokenizers.Encoding object rather than a plain list of ints, which
        # breaks torch.tensor(). encode() always returns a plain Python list.
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        full_ids = full_ids[: self.max_seq_len]
        seq_len = len(full_ids)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        labels = torch.full((seq_len,), -100, dtype=torch.long)

        # Unmask only the tokens that belong to assistant turns.
        # For each assistant turn at position i in the message list:
        #   - render prefix messages[:i] with add_generation_prompt=True
        #     → ends with "<|im_start|>assistant\n", so len(prefix_ids) is
        #       the index of the first assistant content token
        #   - render messages[:i+1] with add_generation_prompt=False
        #     → includes assistant content + <|im_end|> terminator
        #   - unmask tokens in [start, end) clipped to truncated seq length
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            prefix_text = self.tokenizer.apply_chat_template(
                messages[:i],
                tokenize=False,
                add_generation_prompt=True,
            )
            upto_text = self.tokenizer.apply_chat_template(
                messages[: i + 1],
                tokenize=False,
                add_generation_prompt=False,
            )

            start = len(self.tokenizer.encode(prefix_text, add_special_tokens=False))
            end = min(len(self.tokenizer.encode(upto_text, add_special_tokens=False)), seq_len)

            if start < end:
                labels[start:end] = input_ids[start:end]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
