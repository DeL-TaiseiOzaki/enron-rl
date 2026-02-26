"""Analyze reward distributions from rollouts.bin files.

Decodes TrainingBatch msgspec objects and reports reward statistics.
"""

import sys
from pathlib import Path

import msgspec

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prime_rl.transport.types import TrainingBatch


def analyze_rollouts(output_dir: Path) -> None:
    """Analyze all rollout files in the given output directory."""
    rollout_dir = output_dir / "run_default" / "rollouts"
    if not rollout_dir.exists():
        print(f"No rollouts directory found at {rollout_dir}")
        return

    decoder = msgspec.msgpack.Decoder(type=TrainingBatch)

    step_dirs = sorted(rollout_dir.iterdir(), key=lambda p: int(p.name.split("_")[1]))
    print(f"\n{'='*70}")
    print(f"Analyzing: {output_dir.name}")
    print(f"Found {len(step_dirs)} steps")
    print(f"{'='*70}")

    all_rewards = []
    all_advantages = []

    for step_dir in step_dirs:
        bin_file = step_dir / "rollouts.bin"
        if not bin_file.exists():
            continue

        with open(bin_file, "rb") as f:
            batch: TrainingBatch = decoder.decode(f.read())

        step = batch.step
        rewards = [ex.reward for ex in batch.examples if ex.reward is not None]
        advantages = [ex.advantage for ex in batch.examples if ex.advantage is not None]

        if not rewards:
            print(f"  Step {step:3d}: no rewards found")
            continue

        all_rewards.extend(rewards)
        all_advantages.extend(advantages)

        # Categorize rewards
        correct = sum(1 for r in rewards if r > 0.5)       # judge=1.0 + tool bonus
        wrong = sum(1 for r in rewards if r < -0.5)         # judge=-1.0
        neutral = sum(1 for r in rewards if -0.5 <= r <= 0.5)  # judge=0.0 (IDK/timeout)

        mean_r = sum(rewards) / len(rewards)
        min_r = min(rewards)
        max_r = max(rewards)

        print(
            f"  Step {step:3d}: n={len(rewards):4d} | "
            f"mean={mean_r:+.4f} min={min_r:+.4f} max={max_r:+.4f} | "
            f"correct={correct} wrong={wrong} neutral={neutral}"
        )

    if all_rewards:
        print(f"\n--- Overall Summary ({output_dir.name}) ---")
        mean_r = sum(all_rewards) / len(all_rewards)
        print(f"  Total samples: {len(all_rewards)}")
        print(f"  Mean reward: {mean_r:+.4f}")
        print(f"  Min reward: {min(all_rewards):+.4f}")
        print(f"  Max reward: {max(all_rewards):+.4f}")

        # Reward distribution buckets
        buckets = {}
        for r in all_rewards:
            # Round to nearest 0.1
            bucket = round(r, 1)
            buckets[bucket] = buckets.get(bucket, 0) + 1

        print(f"\n  Reward distribution:")
        for bucket in sorted(buckets.keys()):
            count = buckets[bucket]
            pct = count / len(all_rewards) * 100
            bar = "#" * int(pct)
            print(f"    {bucket:+.1f}: {count:5d} ({pct:5.1f}%) {bar}")

        # Key metric: fraction with positive vs negative rewards
        positive = sum(1 for r in all_rewards if r > 0)
        negative = sum(1 for r in all_rewards if r < 0)
        zero = sum(1 for r in all_rewards if r == 0.0)
        print(f"\n  Positive rewards: {positive} ({positive/len(all_rewards)*100:.1f}%)")
        print(f"  Zero rewards:    {zero} ({zero/len(all_rewards)*100:.1f}%)")
        print(f"  Negative rewards: {negative} ({negative/len(all_rewards)*100:.1f}%)")

        if all_advantages:
            mean_a = sum(all_advantages) / len(all_advantages)
            print(f"\n  Mean advantage: {mean_a:+.6f}")
            print(f"  Min advantage: {min(all_advantages):+.4f}")
            print(f"  Max advantage: {max(all_advantages):+.4f}")


if __name__ == "__main__":
    base = Path("outputs")
    for model_dir in ["1.7b", "4b", "8b"]:
        model_path = base / model_dir
        if model_path.exists():
            analyze_rollouts(model_path)
