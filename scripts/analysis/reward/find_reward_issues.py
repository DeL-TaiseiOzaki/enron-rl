#!/usr/bin/env python3
"""Find potential reward issues - cases where reward might be wrong."""

import json
import re
from pathlib import Path


def extract_answer_and_tools(messages: str) -> tuple[str, int, bool]:
    """Extract final answer, tool count, and whether it returned an answer."""
    tool_calls = re.findall(r'<tool_call>', messages)
    tool_count = len(tool_calls)

    # Check for return_final_answer
    has_return = 'return_final_answer' in messages

    # Extract answer
    answer = "NO ANSWER"
    if has_return:
        match = re.search(r'"answer":\s*"([^"]*(?:\\"[^"]*)*)"', messages)
        if match:
            answer = match.group(1).replace('\\"', '"')

    return answer, tool_count, has_return


def main():
    """Find reward anomalies."""
    files = [
        Path("/home/usr_ext_taisei_ozaki_ccoe_toyota/enron-rl/outputs/4b/run_default/wandb/run-20260210_045515-z4zqjeya/files/media/table/samples_1_3c1385a75c61487c1db0.table.json"),
        Path("/home/usr_ext_taisei_ozaki_ccoe_toyota/enron-rl/outputs/4b/run_default/wandb/run-20260210_045515-z4zqjeya/files/media/table/final-samples_39_42d46a151eebaca53366.table.json"),
    ]

    issues = {
        "high_reward_no_answer": [],
        "negative_reward_with_substantive_answer": [],
        "idk_with_positive_reward": [],
        "similar_answers_different_rewards": [],
    }

    all_samples = []

    for file_path in files:
        with open(file_path) as f:
            data = json.load(f)

        for row in data["data"]:
            step, task, example_id, messages, input_ids, reward = row

            # Extract query
            user_match = re.search(r'<\|im_start\|>user\n(.+?)<\|im_end\|>', messages, re.DOTALL)
            query = user_match.group(1).strip() if user_match else "N/A"

            answer, tool_count, has_return = extract_answer_and_tools(messages)

            sample = {
                "example_id": example_id,
                "query": query,
                "answer": answer,
                "tool_count": tool_count,
                "has_return": has_return,
                "reward": reward,
                "file": file_path.name,
            }
            all_samples.append(sample)

            # Check for issues
            # Issue 1: High reward but no answer returned
            if reward > 0.5 and not has_return:
                issues["high_reward_no_answer"].append(sample)

            # Issue 2: Negative reward with substantive answer (not "I don't know")
            if reward < -0.5 and has_return and "don't know" not in answer.lower() and len(answer) > 20:
                issues["negative_reward_with_substantive_answer"].append(sample)

            # Issue 3: "I don't know" with positive correctness reward
            estimated_correctness = reward - min(tool_count * 0.08, 0.8)
            if "don't know" in answer.lower() and estimated_correctness > 0.3:
                issues["idk_with_positive_reward"].append(sample)

    # Print issues
    print("\n" + "="*100)
    print("POTENTIAL REWARD ISSUES")
    print("="*100)

    for issue_type, samples in issues.items():
        if samples:
            print(f"\n{'='*100}")
            print(f"ISSUE: {issue_type.upper().replace('_', ' ')}")
            print(f"Count: {len(samples)}")
            print(f"{'='*100}")

            for sample in samples[:3]:  # Show first 3
                print(f"\nExample ID: {sample['example_id']} | Reward: {sample['reward']:.4f} | Tools: {sample['tool_count']}")
                print(f"Query: {sample['query'][:150]}")
                print(f"Answer: {sample['answer'][:200]}")
                print("-" * 100)

    # Look for duplicate example_ids with different rewards
    print(f"\n{'='*100}")
    print("CHECKING FOR DUPLICATE EXAMPLES WITH DIFFERENT REWARDS")
    print(f"{'='*100}")

    example_groups = {}
    for sample in all_samples:
        eid = sample["example_id"]
        if eid not in example_groups:
            example_groups[eid] = []
        example_groups[eid].append(sample)

    duplicates_with_diff_rewards = []
    for eid, samples in example_groups.items():
        if len(samples) > 1:
            rewards = [s["reward"] for s in samples]
            if len(set(rewards)) > 1:  # Different rewards
                duplicates_with_diff_rewards.append((eid, samples))

    if duplicates_with_diff_rewards:
        print(f"\nFound {len(duplicates_with_diff_rewards)} examples with different rewards:")
        for eid, samples in duplicates_with_diff_rewards[:3]:
            print(f"\nExample ID: {eid}")
            for s in samples:
                print(f"  Reward: {s['reward']:.4f} | Tools: {s['tool_count']} | File: {s['file']}")
                print(f"  Answer: {s['answer'][:100]}")
    else:
        print("\nNo duplicates with different rewards found.")

    # Show reward distribution
    print(f"\n{'='*100}")
    print("REWARD DISTRIBUTION SUMMARY")
    print(f"{'='*100}")

    rewards = [s["reward"] for s in all_samples]
    print(f"Total samples: {len(all_samples)}")
    print(f"Min reward: {min(rewards):.4f}")
    print(f"Max reward: {max(rewards):.4f}")
    print(f"Avg reward: {sum(rewards)/len(rewards):.4f}")

    # Count by category
    positive = len([r for r in rewards if r > 0.5])
    negative = len([r for r in rewards if r < -0.5])
    neutral = len([r for r in rewards if -0.5 <= r <= 0.5])

    print(f"\nPositive (>0.5): {positive}")
    print(f"Negative (<-0.5): {negative}")
    print(f"Neutral (-0.5 to 0.5): {neutral}")

    # Show breakdown of answers
    print(f"\n{'='*100}")
    print("ANSWER ANALYSIS")
    print(f"{'='*100}")

    no_return = len([s for s in all_samples if not s["has_return"]])
    idk = len([s for s in all_samples if "don't know" in s["answer"].lower()])
    substantive = len([s for s in all_samples if s["has_return"] and "don't know" not in s["answer"].lower()])

    print(f"No return_final_answer: {no_return}")
    print(f"'I don't know': {idk}")
    print(f"Substantive answer: {substantive}")


if __name__ == "__main__":
    main()
