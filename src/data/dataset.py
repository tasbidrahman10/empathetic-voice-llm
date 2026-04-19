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

        # Tokenize the full conversation (returns a list of int token IDs)
        full_ids: list[int] = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        # Truncate to max_seq_len from the right
        full_ids = full_ids[: self.max_seq_len]
        seq_len = len(full_ids)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        labels = torch.full((seq_len,), -100, dtype=torch.long)

        # Unmask only the tokens that belong to assistant turns.
        # Strategy: for each assistant turn at position i in the message list,
        #   - tokenize prefix messages[:i] with add_generation_prompt=True
        #     → this ends with the "<|im_start|>assistant\n" header tokens,
        #       so len(prefix_ids) is the index of the first assistant content token
        #   - tokenize messages[:i+1] with add_generation_prompt=False
        #     → includes the assistant content + <|im_end|> terminator
        #   - unmask tokens in [start, end) within the (possibly truncated) sequence
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            prefix_ids: list[int] = self.tokenizer.apply_chat_template(
                messages[:i],
                tokenize=True,
                add_generation_prompt=True,
            )
            upto_ids: list[int] = self.tokenizer.apply_chat_template(
                messages[: i + 1],
                tokenize=True,
                add_generation_prompt=False,
            )

            start = len(prefix_ids)
            end = min(len(upto_ids), seq_len)

            if start < end:
                labels[start:end] = input_ids[start:end]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
