"""Visualize RL training metrics from trainer and orchestrator logs."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("outputs/1.7b")
TRAINER_LOG = OUTPUT_DIR / "logs/trainer/rank_0.log"
ORCHESTRATOR_LOG = OUTPUT_DIR / "run_default/logs/orchestrator.log"
PLOT_OUTPUT = OUTPUT_DIR / "training_curves.png"

# ANSI escape code removal
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_trainer_log(path: Path) -> dict[str, list[float]]:
    """Parse trainer SUCCESS lines for metrics."""
    pattern = re.compile(
        r"Step (\d+) \| Time: ([0-9.]+)s \| Loss: ([0-9.e+-]+) \| "
        r"Entropy: ([0-9.e+-]+) \| Mismatch KL: ([0-9.e+-]+) \| "
        r"Grad\. Norm: ([0-9.e+-]+) \| LR: ([0-9.e+-]+) \| "
        r"Throughput: (\d+) tokens/s \| MFU: ([0-9.]+)% \| "
        r"Peak Mem\.: ([0-9.]+) GiB"
    )
    data: dict[str, list[float]] = {
        "step": [], "time": [], "loss": [], "entropy": [],
        "mismatch_kl": [], "grad_norm": [], "lr": [],
        "throughput": [], "mfu": [], "peak_mem": [],
    }
    text = ANSI_RE.sub("", path.read_text())
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            data["step"].append(int(m.group(1)))
            data["time"].append(float(m.group(2)))
            data["loss"].append(float(m.group(3)))
            data["entropy"].append(float(m.group(4)))
            data["mismatch_kl"].append(float(m.group(5)))
            data["grad_norm"].append(float(m.group(6)))
            data["lr"].append(float(m.group(7)))
            data["throughput"].append(float(m.group(8)))
            data["mfu"].append(float(m.group(9)))
            data["peak_mem"].append(float(m.group(10)))
    return data


def parse_orchestrator_log(path: Path) -> dict[str, list[float]]:
    """Parse orchestrator SUCCESS lines for reward and seq length."""
    pattern = re.compile(
        r"Step (\d+) \| Time: ([0-9.]+)s \| Reward: ([0-9.e+-]+) \| "
        r"Throughput: ([0-9.]+) tokens/s \| "
        r"Seq\. Length: ([0-9.]+) tokens/sample"
    )
    data: dict[str, list[float]] = {
        "step": [], "time": [], "reward": [],
        "throughput": [], "seq_length": [],
    }
    text = ANSI_RE.sub("", path.read_text())
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            data["step"].append(int(m.group(1)))
            data["time"].append(float(m.group(2)))
            data["reward"].append(float(m.group(3)))
            data["throughput"].append(float(m.group(4)))
            data["seq_length"].append(float(m.group(5)))
    return data


def moving_average(values: list[float], window: int = 5) -> np.ndarray:
    """Compute simple moving average."""
    arr = np.array(values)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    # Use 'valid' then pad with NaN at the start
    ma = np.convolve(arr, kernel, mode="valid")
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, ma])


def plot_training(trainer: dict, orch: dict, output_path: Path) -> None:
    """Create a comprehensive training dashboard."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle(
        "Qwen3-1.7B RL Training (LoRA r=16, lr=5e-6, batch=32, 79 steps / 1 epoch)",
        fontsize=16, fontweight="bold", y=0.98,
    )

    t_steps = np.array(trainer["step"])
    o_steps = np.array(orch["step"])

    # --- Row 1: Core RL metrics ---

    # 1. Reward (orchestrator)
    ax = axes[0, 0]
    ax.plot(o_steps, orch["reward"], alpha=0.4, color="tab:blue", linewidth=0.8)
    ma = moving_average(orch["reward"], window=10)
    ax.plot(o_steps, ma, color="tab:blue", linewidth=2, label="MA(10)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Mean Reward per Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Policy Loss (trainer)
    ax = axes[0, 1]
    ax.plot(t_steps, trainer["loss"], alpha=0.4, color="tab:red", linewidth=0.8)
    ma = moving_average(trainer["loss"], window=10)
    ax.plot(t_steps, ma, color="tab:red", linewidth=2, label="MA(10)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Policy Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Entropy
    ax = axes[0, 2]
    ax.plot(t_steps, trainer["entropy"], alpha=0.4, color="tab:green", linewidth=0.8)
    ma = moving_average(trainer["entropy"], window=10)
    ax.plot(t_steps, ma, color="tab:green", linewidth=2, label="MA(10)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Row 2: Training dynamics ---

    # 4. KL Divergence
    ax = axes[1, 0]
    ax.plot(t_steps, trainer["mismatch_kl"], alpha=0.4, color="tab:purple", linewidth=0.8)
    ma = moving_average(trainer["mismatch_kl"], window=10)
    ax.plot(t_steps, ma, color="tab:purple", linewidth=2, label="MA(10)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mismatch KL")
    ax.set_title("KL Divergence (Policy vs Reference)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Gradient Norm
    ax = axes[1, 1]
    ax.plot(t_steps, trainer["grad_norm"], alpha=0.4, color="tab:orange", linewidth=0.8)
    ma = moving_average(trainer["grad_norm"], window=10)
    ax.plot(t_steps, ma, color="tab:orange", linewidth=2, label="MA(10)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Sequence Length (orchestrator)
    ax = axes[1, 2]
    ax.plot(o_steps, orch["seq_length"], alpha=0.4, color="tab:cyan", linewidth=0.8)
    ma = moving_average(orch["seq_length"], window=10)
    ax.plot(o_steps, ma, color="tab:cyan", linewidth=2, label="MA(10)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens / Sample")
    ax.set_title("Average Sequence Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Row 3: Performance ---

    # 7. Trainer Throughput
    ax = axes[2, 0]
    ax.plot(t_steps, trainer["throughput"], alpha=0.6, color="tab:brown", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens/s")
    ax.set_title("Trainer Throughput")
    ax.grid(True, alpha=0.3)

    # 8. Step Time (trainer)
    ax = axes[2, 1]
    ax.plot(t_steps, trainer["time"], alpha=0.6, color="tab:gray", linewidth=1)
    ma = moving_average(trainer["time"], window=10)
    ax.plot(t_steps, ma, color="tab:gray", linewidth=2, label="MA(10)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Seconds")
    ax.set_title("Step Time (Trainer)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9. Reward distribution histogram
    ax = axes[2, 2]
    rewards = np.array(orch["reward"])
    ax.hist(rewards, bins=25, color="tab:blue", alpha=0.7, edgecolor="black")
    ax.axvline(x=rewards.mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {rewards.mean():.3f}")
    ax.axvline(x=np.median(rewards), color="orange", linestyle="--", linewidth=2,
               label=f"Median: {np.median(rewards):.3f}")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)

    # Summary stats
    print("\n=== Training Summary ===")
    print(f"Total steps: {len(t_steps)} (trainer), {len(o_steps)} (orchestrator)")
    print(f"Reward: mean={rewards.mean():.4f}, std={rewards.std():.4f}, "
          f"min={rewards.min():.4f}, max={rewards.max():.4f}")
    print(f"Final 10 steps reward mean: {rewards[-10:].mean():.4f}")
    print(f"First 10 steps reward mean: {rewards[:10].mean():.4f}")

    losses = np.array(trainer["loss"])
    print(f"Loss: mean={losses.mean():.6f}, std={losses.std():.6f}")
    print(f"Final 10 steps loss mean: {losses[-10:].mean():.6f}")

    entropies = np.array(trainer["entropy"])
    print(f"Entropy: start={entropies[0]:.4f}, end={entropies[-1]:.4f}, "
          f"change={entropies[-1] - entropies[0]:+.4f}")

    kls = np.array(trainer["mismatch_kl"])
    print(f"KL: start={kls[0]:.6f}, end={kls[-1]:.6f}, "
          f"change={kls[-1] - kls[0]:+.6f}")

    grad_norms = np.array(trainer["grad_norm"])
    print(f"Grad norm: mean={grad_norms.mean():.4f}, max={grad_norms.max():.4f}")

    total_time_trainer = np.array(trainer["time"]).sum()
    total_time_orch = np.array(orch["time"]).sum()
    print(f"Total trainer time: {total_time_trainer/60:.1f} min")
    print(f"Total orchestrator time: {total_time_orch/60:.1f} min")


if __name__ == "__main__":
    print("Parsing trainer log...")
    trainer_data = parse_trainer_log(TRAINER_LOG)
    print(f"  Found {len(trainer_data['step'])} steps")

    print("Parsing orchestrator log...")
    orch_data = parse_orchestrator_log(ORCHESTRATOR_LOG)
    print(f"  Found {len(orch_data['step'])} steps")

    plot_training(trainer_data, orch_data, PLOT_OUTPUT)
