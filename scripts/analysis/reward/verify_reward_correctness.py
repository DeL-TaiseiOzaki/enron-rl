#!/usr/bin/env python3
"""Verify reward correctness by examining tool results in detail."""

import json
import re
from pathlib import Path


def analyze_negative_reward_sample(messages: str, example_id: int, reward: float):
    """Analyze a negative reward sample to see if answer was actually correct."""
    print(f"\n{'='*100}")
    print(f"EXAMPLE ID: {example_id} | REWARD: {reward:.4f}")
    print(f"{'='*100}")

    # Extract query
    user_match = re.search(r'<\|im_start\|>user\n(.+?)<\|im_end\|>', messages, re.DOTALL)
    if user_match:
        query = user_match.group(1).strip()
        print(f"\n❓ QUERY:\n{query}\n")

    # Extract final answer
    answer_match = re.search(r'"answer":\s*"([^"]*(?:\\"[^"]*)*)"', messages)
    if answer_match:
        answer = answer_match.group(1).replace('\\"', '"')
        print(f"🤖 MODEL ANSWER:\n{answer}\n")

    # Extract all tool results
    tool_results = re.findall(
        r'<\|im_start\|>tool\n(.*?)<\|im_end\|>',
        messages,
        re.DOTALL
    )

    print(f"📊 TOOL CALLS: {len(tool_results)}")

    # Show tool results
    for i, result in enumerate(tool_results, 1):
        print(f"\n🔧 TOOL RESULT {i}:")
        # Try to parse as JSON
        try:
            result_clean = result.strip()
            if result_clean.startswith('{'):
                result_data = json.loads(result_clean)
                if 'results' in result_data:
                    print(f"   Found {len(result_data['results'])} email snippets")
                    for j, snippet in enumerate(result_data['results'][:2], 1):  # First 2
                        print(f"\n   Snippet {j}:")
                        if 'subject' in snippet:
                            print(f"   Subject: {snippet['subject']}")
                        if 'snippet' in snippet:
                            print(f"   Snippet: {snippet['snippet'][:200]}")
                elif 'subject' in result_data:
                    # Full email read
                    print(f"   Subject: {result_data.get('subject', 'N/A')}")
                    body = result_data.get('body', '')
                    print(f"   Body preview: {body[:300]}")
        except:
            print(f"   {result[:300]}")

    print(f"\n{'='*100}")
    print("❓ VERDICT: Does the model answer match the email evidence?")
    print("="*100)


def main():
    """Verify negative reward samples."""
    files = [
        Path("/home/usr_ext_taisei_ozaki_ccoe_toyota/enron-rl/outputs/4b/run_default/wandb/run-20260210_045515-z4zqjeya/files/media/table/samples_1_3c1385a75c61487c1db0.table.json"),
        Path("/home/usr_ext_taisei_ozaki_ccoe_toyota/enron-rl/outputs/4b/run_default/wandb/run-20260210_045515-z4zqjeya/files/media/table/final-samples_39_42d46a151eebaca53366.table.json"),
    ]

    # Focus on negative reward samples with substantive answers
    negative_samples = []

    for file_path in files:
        with open(file_path) as f:
            data = json.load(f)

        for row in data["data"]:
            step, task, example_id, messages, input_ids, reward = row

            if reward < -0.5:
                # Check if it has a substantive answer
                has_return = 'return_final_answer' in messages
                if has_return:
                    answer_match = re.search(r'"answer":\s*"([^"]*(?:\\"[^"]*)*)"', messages)
                    if answer_match:
                        answer = answer_match.group(1).replace('\\"', '"')
                        if "don't know" not in answer.lower() and len(answer) > 20:
                            negative_samples.append((example_id, reward, messages))

    # Remove duplicates by example_id
    seen = set()
    unique_samples = []
    for eid, reward, messages in negative_samples:
        if eid not in seen:
            seen.add(eid)
            unique_samples.append((eid, reward, messages))

    print(f"Found {len(unique_samples)} unique negative-reward samples with substantive answers")

    # Analyze first 3
    for example_id, reward, messages in unique_samples[:3]:
        analyze_negative_reward_sample(messages, example_id, reward)
        print("\n" + "="*100)
        input("Press Enter to continue to next sample...")


if __name__ == "__main__":
    main()
