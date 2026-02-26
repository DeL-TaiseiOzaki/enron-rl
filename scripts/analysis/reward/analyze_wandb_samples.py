#!/usr/bin/env python3
"""Analyze W&B sample tables to understand reward vs answer quality relationship."""

import json
import re
from pathlib import Path
from typing import Any


def extract_user_query(messages_text: str) -> str:
    """Extract the user's query from messages."""
    # Find the first user message after system
    match = re.search(r'<\|im_start\|>user\n(.+?)<\|im_end\|>', messages_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "NOT FOUND"


def extract_final_answer(messages_text: str) -> str:
    """Extract the final answer from return_final_answer tool call."""
    # Look for return_final_answer function call
    match = re.search(
        r'<tool_call>\s*\{.*?"name":\s*"return_final_answer".*?"arguments":\s*({[^}]*?(?:"answer":\s*"[^"]*")[^}]*?})',
        messages_text,
        re.DOTALL
    )
    if match:
        try:
            args_json = match.group(1)
            # Handle escaped quotes
            args_json = args_json.replace('\\"', '"').replace('\\n', '\n')
            args = json.loads(args_json)
            return args.get('answer', 'NO ANSWER IN ARGS')
        except json.JSONDecodeError:
            # Try to extract just the answer field
            answer_match = re.search(r'"answer":\s*"([^"]*)"', match.group(1))
            if answer_match:
                return answer_match.group(1)
            return "PARSE ERROR"

    # Check if there's any assistant response at all
    if '<|im_start|>assistant' in messages_text:
        return "NO return_final_answer CALL"
    return "NO ASSISTANT RESPONSE"


def extract_reference_answer(messages_text: str) -> str:
    """Extract reference answer from system prompt or metadata."""
    # Some examples might have reference in system prompt or as metadata
    # For now, return placeholder
    return "N/A (not in messages)"


def analyze_sample(data_row: list) -> dict[str, Any]:
    """Analyze a single sample row."""
    step, task, example_id, messages, input_ids, reward = data_row

    query = extract_user_query(messages)
    answer = extract_final_answer(messages)
    reference = extract_reference_answer(messages)

    return {
        "step": step,
        "task": task,
        "example_id": example_id,
        "query": query[:200] + "..." if len(query) > 200 else query,
        "answer": answer[:300] + "..." if len(answer) > 300 else answer,
        "reference": reference,
        "reward": reward,
        "full_messages_length": len(messages),
    }


def categorize_by_reward(samples: list[dict]) -> dict[str, list[dict]]:
    """Categorize samples by reward ranges."""
    categories = {
        "high_positive": [],  # ~1.08
        "medium_positive": [],  # 0.5-0.9
        "low_positive": [],  # 0.1-0.5
        "near_zero": [],  # -0.1 to 0.1
        "negative": [],  # < -0.5
    }

    for sample in samples:
        reward = sample["reward"]
        if reward >= 1.0:
            categories["high_positive"].append(sample)
        elif reward >= 0.5:
            categories["medium_positive"].append(sample)
        elif reward >= 0.1:
            categories["low_positive"].append(sample)
        elif reward >= -0.1:
            categories["near_zero"].append(sample)
        else:
            categories["negative"].append(sample)

    return categories


def main():
    """Main analysis."""
    files = [
        Path("/home/usr_ext_taisei_ozaki_ccoe_toyota/enron-rl/outputs/4b/run_default/wandb/run-20260210_045515-z4zqjeya/files/media/table/samples_1_3c1385a75c61487c1db0.table.json"),
        Path("/home/usr_ext_taisei_ozaki_ccoe_toyota/enron-rl/outputs/4b/run_default/wandb/run-20260210_045515-z4zqjeya/files/media/table/final-samples_39_42d46a151eebaca53366.table.json"),
    ]

    all_samples = []

    for file_path in files:
        print(f"\n{'='*80}")
        print(f"Analyzing: {file_path.name}")
        print(f"{'='*80}")

        with open(file_path) as f:
            data = json.load(f)

        print(f"Total samples in file: {len(data['data'])}")

        for row in data["data"][:10]:  # First 10 samples per file
            sample = analyze_sample(row)
            all_samples.append(sample)

    # Categorize
    categories = categorize_by_reward(all_samples)

    print("\n" + "="*80)
    print("REWARD DISTRIBUTION")
    print("="*80)
    for cat_name, samples in categories.items():
        print(f"{cat_name}: {len(samples)} samples")

    # Show examples from each category
    print("\n" + "="*80)
    print("SAMPLE EXAMPLES BY REWARD CATEGORY")
    print("="*80)

    for cat_name, samples in categories.items():
        if samples:
            print(f"\n{'='*80}")
            print(f"CATEGORY: {cat_name.upper()}")
            print(f"{'='*80}")
            # Show first 2 from each category
            for sample in samples[:2]:
                print(f"\nStep: {sample['step']} | Example ID: {sample['example_id']} | Reward: {sample['reward']:.4f}")
                print(f"\nQuery:\n{sample['query']}")
                print(f"\nModel Answer:\n{sample['answer']}")
                print(f"\nReference:\n{sample['reference']}")
                print("-" * 80)


if __name__ == "__main__":
    main()
