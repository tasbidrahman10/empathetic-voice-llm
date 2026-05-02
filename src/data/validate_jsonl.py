#!/usr/bin/env python3
"""Validate processed EFSM JSONL files before training.

This is intentionally lightweight so it can be run in Colab/Kaggle before any
GPU work. It catches the main failure mode where assistant targets accidentally
copy the user/context prompt, producing unrealistically tiny train/eval loss.
"""

import argparse
import json
from pathlib import Path


def strip_emotion_tag(text: str) -> str:
    if text.startswith("[emotion:") and "] " in text:
        return text.split("] ", 1)[1].strip()
    return text.strip()


def validate_file(path: Path, max_copy_rate: float) -> None:
    conversations = 0
    assistant_turns = 0
    copied_turns = 0
    empty_turns = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            messages = json.loads(line)
            conversations += 1
            previous_user = None

            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "").strip()
                if not content:
                    empty_turns += 1
                if role == "user":
                    previous_user = strip_emotion_tag(content)
                elif role == "assistant":
                    assistant_turns += 1
                    if previous_user and content == previous_user:
                        copied_turns += 1

            if not any(m.get("role") == "assistant" for m in messages):
                raise ValueError(f"{path}:{line_no} has no assistant turn")

    copy_rate = copied_turns / assistant_turns if assistant_turns else 0.0
    print(f"{path}")
    print(f"  conversations       : {conversations}")
    print(f"  assistant turns     : {assistant_turns}")
    print(f"  empty turns         : {empty_turns}")
    print(f"  exact user copies   : {copied_turns} ({copy_rate:.2%})")

    if copy_rate > max_copy_rate:
        raise ValueError(
            f"{path} failed: assistant/user exact-copy rate {copy_rate:.2%} "
            f"exceeds allowed {max_copy_rate:.2%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max-copy-rate", type=float, default=0.05)
    args = parser.parse_args()

    for path in args.paths:
        validate_file(path, args.max_copy_rate)


if __name__ == "__main__":
    main()
