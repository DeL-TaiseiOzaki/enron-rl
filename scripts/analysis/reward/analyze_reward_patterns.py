#!/usr/bin/env python3
"""
Comprehensive analysis of reward patterns.
Since messages don't include tool responses, focus on:
1. Reward components (correctness + tool_count)
2. Answer patterns
3. Identifying potential issues
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def extract_info(messages: str) -> dict:
    """Extract key information from messages."""
    # Query
    user_match = re.search(r'<\|im_start\|>user\n(.+?)<\|im_end\|>', messages, re.DOTALL)
    query = user_match.group(1).strip() if user_match else "N/A"

    # Tool calls
    tool_calls = len(re.findall(r'<tool_call>', messages))

    # Has return_final_answer
    has_return = 'return_final_answer' in messages

    # Extract answer
    answer = "NO ANSWER"
    if has_return:
        match = re.search(r'"answer":\s*"([^"]*(?:\\"[^"]*)*)"', messages)
        if match:
            answer = match.group(1).replace('\\"', '"')

    return {
        "query": query,
        "tool_calls": tool_calls,
        "has_return": has_return,
        "answer": answer,
    }


def categorize_answer(answer: str) -> str:
    """Categorize answer type."""
    if answer == "NO ANSWER":
        return "no_return"
    elif "don't know" in answer.lower() or "do not know" in answer.lower():
        return "idk"
    elif len(answer) < 10:
        return "very_short"
    else:
        return "substantive"


def main():
    """Main analysis."""
    files = [
        Path("/home/usr_ext_taisei_ozaki_ccoe_toyota/enron-rl/outputs/4b/run_default/wandb/run-20260210_045515-z4zqjeya/files/media/table/samples_1_3c1385a75c61487c1db0.table.json"),
        Path("/home/usr_ext_taisei_ozaki_ccoe_toyota/enron-rl/outputs/4b/run_default/wandb/run-20260210_045515-z4zqjeya/files/media/table/final-samples_39_42d46a151eebaca53366.table.json"),
    ]

    all_samples = []

    for file_path in files:
        with open(file_path) as f:
            data = json.load(f)

        for row in data["data"]:
            step, task, example_id, messages, input_ids, reward = row
            info = extract_info(messages)

            sample = {
                "step": step,
                "example_id": example_id,
                "reward": reward,
                "tool_calls": info["tool_calls"],
                "query": info["query"],
                "answer": info["answer"],
                "answer_type": categorize_answer(info["answer"]),
                "file": file_path.name,
            }
            all_samples.append(sample)

    print("\n" + "="*100)
    print("COMPREHENSIVE REWARD ANALYSIS")
    print("="*100)

    print(f"\nTotal samples: {len(all_samples)}")

    # Reward components
    print(f"\n{'='*100}")
    print("REWARD COMPONENT ANALYSIS")
    print(f"{'='*100}")

    for sample in all_samples:
        tool_reward = min(sample["tool_calls"] * 0.08, 0.8)
        correctness = sample["reward"] - tool_reward
        sample["estimated_tool_reward"] = tool_reward
        sample["estimated_correctness"] = correctness

    # Group by answer type and reward
    print(f"\n{'='*100}")
    print("ANSWER TYPE vs REWARD")
    print(f"{'='*100}")

    by_type = defaultdict(list)
    for sample in all_samples:
        by_type[sample["answer_type"]].append(sample)

    for answer_type, samples in sorted(by_type.items()):
        rewards = [s["estimated_correctness"] for s in samples]
        avg_correctness = sum(rewards) / len(rewards)
        print(f"\n{answer_type.upper()}: {len(samples)} samples")
        print(f"  Avg correctness reward: {avg_correctness:.4f}")
        print(f"  Range: {min(rewards):.4f} to {max(rewards):.4f}")

    # Show examples from each category
    print(f"\n{'='*100}")
    print("DETAILED EXAMPLES BY CATEGORY")
    print(f"{'='*100}")

    # 1. High positive (correct)
    high_positive = [s for s in all_samples if s["estimated_correctness"] > 0.5]
    print(f"\n{'='*100}")
    print(f"✅ HIGH POSITIVE CORRECTNESS (>0.5): {len(high_positive)} samples")
    print(f"{'='*100}")
    for sample in high_positive[:2]:
        print(f"\nExample ID: {sample['example_id']} | Reward: {sample['reward']:.4f} | Correctness: {sample['estimated_correctness']:.4f}")
        print(f"Tools: {sample['tool_calls']} | Answer Type: {sample['answer_type']}")
        print(f"Query: {sample['query'][:120]}")
        print(f"Answer: {sample['answer'][:150]}")

    # 2. High negative (incorrect)
    high_negative = [s for s in all_samples if s["estimated_correctness"] < -0.5]
    print(f"\n{'='*100}")
    print(f"❌ HIGH NEGATIVE CORRECTNESS (<-0.5): {len(high_negative)} samples")
    print(f"{'='*100}")
    for sample in high_negative[:3]:
        print(f"\nExample ID: {sample['example_id']} | Reward: {sample['reward']:.4f} | Correctness: {sample['estimated_correctness']:.4f}")
        print(f"Tools: {sample['tool_calls']} | Answer Type: {sample['answer_type']}")
        print(f"Query: {sample['query'][:120]}")
        print(f"Answer: {sample['answer'][:150]}")

    # 3. Near zero (neutral)
    near_zero = [s for s in all_samples if -0.5 <= s["estimated_correctness"] <= 0.1]
    print(f"\n{'='*100}")
    print(f"⚪ NEAR-ZERO CORRECTNESS (-0.5 to 0.1): {len(near_zero)} samples")
    print(f"{'='*100}")
    for sample in near_zero[:3]:
        print(f"\nExample ID: {sample['example_id']} | Reward: {sample['reward']:.4f} | Correctness: {sample['estimated_correctness']:.4f}")
        print(f"Tools: {sample['tool_calls']} | Answer Type: {sample['answer_type']}")
        print(f"Query: {sample['query'][:120]}")
        print(f"Answer: {sample['answer'][:150]}")

    # KEY FINDINGS
    print(f"\n{'='*100}")
    print("🔍 KEY FINDINGS")
    print(f"{'='*100}")

    # Finding 1: substantive answers with negative correctness
    substantive_negative = [s for s in all_samples
                           if s["answer_type"] == "substantive" and s["estimated_correctness"] < -0.5]
    print(f"\n1. Substantive answers with NEGATIVE correctness: {len(substantive_negative)}")
    print("   → These answers look reasonable but got marked as incorrect.")
    print("   → Possible reasons: factual errors, hallucination, or judge error.")

    # Finding 2: "I don't know" with tool count reward
    idk_samples = [s for s in all_samples if s["answer_type"] == "idk"]
    if idk_samples:
        avg_idk_total = sum(s["reward"] for s in idk_samples) / len(idk_samples)
        avg_idk_tools = sum(s["tool_calls"] for s in idk_samples) / len(idk_samples)
        print(f"\n2. 'I don't know' answers: {len(idk_samples)}")
        print(f"   → Avg total reward: {avg_idk_total:.4f}")
        print(f"   → Avg tools used: {avg_idk_tools:.2f}")
        print("   → These get neutral/negative correctness but positive tool_count_reward.")
        print("   → This creates perverse incentive: use many tools then say 'I don't know'.")

    # Finding 3: No return samples
    no_return = [s for s in all_samples if s["answer_type"] == "no_return"]
    if no_return:
        avg_no_return = sum(s["estimated_correctness"] for s in no_return) / len(no_return)
        print(f"\n3. No return_final_answer: {len(no_return)}")
        print(f"   → Avg estimated correctness: {avg_no_return:.4f}")
        print("   → Model failed to complete the task.")

    # Comparison: same example_id, different rewards
    print(f"\n4. Same query, different attempts:")
    by_example = defaultdict(list)
    for sample in all_samples:
        by_example[sample["example_id"]].append(sample)

    varied_rewards = []
    for eid, samples in by_example.items():
        if len(samples) > 1:
            rewards = [s["estimated_correctness"] for s in samples]
            if len(set([round(r, 2) for r in rewards])) > 1:
                varied_rewards.append((eid, samples))

    if varied_rewards:
        print(f"   Found {len(varied_rewards)} examples with different correctness scores:")
        for eid, samples in varied_rewards[:2]:
            print(f"\n   Example ID {eid}:")
            for s in samples:
                print(f"      Correctness: {s['estimated_correctness']:.4f} | Tools: {s['tool_calls']} | Answer: {s['answer'][:80]}")


if __name__ == "__main__":
    main()
