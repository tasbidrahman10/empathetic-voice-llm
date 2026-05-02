#!/usr/bin/env python3
"""
Preprocess facebook/empathetic_dialogues into ChatML-formatted JSONL files.

Loads the HuggingFace dataset, groups utterances by conversation ID, formats
each conversation as a ChatML message list with emotion-tagged user turns, then
splits 80/10/10 stratified by emotion and saves to data/{train,val,test}.jsonl.

Usage:
    python src/data/preprocess_empathetic.py --config configs/config.yaml
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """Normalize EmpatheticDialogues placeholders without changing meaning."""
    return text.replace("_comma_", ",").strip()


def strip_emotion_tag(text: str) -> str:
    if text.startswith("[emotion:") and "] " in text:
        return text.split("] ", 1)[1].strip()
    return text.strip()


def build_conversations(dataset_split, system_prompt: str) -> list[dict]:
    """Group utterances by conv_id and build ChatML message lists.

    EmpatheticDialogues alternates speakers: even utterance_idx = seeker (user),
    odd utterance_idx = supporter (assistant). The 'context' field is the emotion
    label, identical across all utterances in a conversation.
    """
    conv_utterances: dict[str, list] = defaultdict(list)
    conv_emotion: dict[str, str] = {}
    conv_prompt: dict[str, str] = {}

    for row in dataset_split:
        cid = row["conv_id"]
        conv_utterances[cid].append(row)
        conv_emotion[cid] = row["context"]
        conv_prompt[cid] = clean_text(row.get("prompt", ""))

    conversations = []
    for cid, utterances in conv_utterances.items():
        utterances.sort(key=lambda r: int(r["utterance_idx"]))
        emotion = conv_emotion[cid]

        messages = [{"role": "system", "content": system_prompt}]
        for i, utt in enumerate(utterances):
            text = clean_text(utt["utterance"])
            if not text:
                continue
            if i % 2 == 0:
                # seeker (user): label with emotion tag
                messages.append({
                    "role": "user",
                    "content": f"[emotion: {emotion}] {text}",
                })
            else:
                # supporter (assistant)
                messages.append({
                    "role": "assistant",
                    "content": text,
                })

        roles = {m["role"] for m in messages}
        if "user" in roles and "assistant" in roles:
            conversations.append({
                "conv_id": cid,
                "emotion": emotion,
                "prompt": conv_prompt.get(cid, ""),
                "messages": messages,
            })

    return conversations


def validate_conversations(conversations: list[dict], max_copy_rate: float = 0.05) -> None:
    """Catch accidental target leakage before expensive fine-tuning starts."""
    total_assistant_turns = 0
    copied_from_previous_user = 0

    for conv in conversations:
        previous_user = None
        for msg in conv["messages"]:
            role = msg["role"]
            content = msg["content"].strip()
            if role == "user":
                previous_user = strip_emotion_tag(content)
            elif role == "assistant":
                total_assistant_turns += 1
                if previous_user and content == previous_user:
                    copied_from_previous_user += 1

    copy_rate = copied_from_previous_user / total_assistant_turns if total_assistant_turns else 0.0
    print(
        f"Leakage check: {copied_from_previous_user}/{total_assistant_turns} "
        f"assistant turns exactly copy the previous user turn ({copy_rate:.2%})"
    )
    if copy_rate > max_copy_rate:
        raise ValueError(
            "Preprocessing failed leakage check: too many assistant turns are "
            "exact copies of the preceding user turn."
        )


def save_jsonl(conversations: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv["messages"], ensure_ascii=False) + "\n")


def print_stats(split_name: str, conversations: list[dict]) -> None:
    emotions = [c["emotion"] for c in conversations]
    emotion_counts = Counter(emotions)
    turn_counts = [
        len([m for m in c["messages"] if m["role"] in ("user", "assistant")])
        for c in conversations
    ]
    avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0

    print(f"\n{split_name}: {len(conversations)} conversations")
    print(f"  Avg turns per conversation: {avg_turns:.1f}")
    print(f"  Emotion distribution (top 10):")
    for emotion, count in emotion_counts.most_common(10):
        print(f"    {emotion:<25} {count:>5}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    system_prompt = cfg["system_prompt"].strip()

    print("Loading facebook/empathetic_dialogues (train split) ...")
    ds = load_dataset(data_cfg["dataset_id"], split="train", trust_remote_code=True)
    print(f"Raw utterance rows: {len(ds)}")

    conversations = build_conversations(ds, system_prompt)
    print(f"Total conversations built: {len(conversations)}")
    validate_conversations(conversations)

    emotions = [c["emotion"] for c in conversations]
    total_test_frac = data_cfg["val_split_ratio"] + data_cfg["test_split_ratio"]

    train_convs, temp_convs, _, temp_emotions = train_test_split(
        conversations,
        emotions,
        test_size=total_test_frac,
        random_state=42,
        stratify=emotions,
    )

    # Split temp 50/50 into val and test (each becomes 10% of total)
    relative_test = data_cfg["test_split_ratio"] / total_test_frac
    val_convs, test_convs, _, _ = train_test_split(
        temp_convs,
        temp_emotions,
        test_size=relative_test,
        random_state=42,
        stratify=temp_emotions,
    )

    print_stats("Train", train_convs)
    print_stats("Val  ", val_convs)
    print_stats("Test ", test_convs)

    save_jsonl(train_convs, data_cfg["train_path"])
    save_jsonl(val_convs, data_cfg["val_path"])
    save_jsonl(test_convs, data_cfg["test_path"])

    print(f"\nSaved:")
    print(f"  {data_cfg['train_path']}  ({len(train_convs)} rows)")
    print(f"  {data_cfg['val_path']}  ({len(val_convs)} rows)")
    print(f"  {data_cfg['test_path']}  ({len(test_convs)} rows)")


if __name__ == "__main__":
    main()
