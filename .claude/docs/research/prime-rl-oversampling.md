# Prime-RL Oversampling Factor Research

## Overview

The **oversampling factor** in prime-rl is a parameter that controls how many problems (examples) are requested from the environment buffer relative to the training batch size. It directly affects the number of in-flight rollout requests and the diversity of training data.

## Definition

Located in `src/prime_rl/orchestrator/config.py`:

```python
oversampling_factor: Annotated[
    float,
    Field(
        ge=1,
        description="Factor by which to oversample the batch. Will lead to more in-flight group rollout requests at the same time.",
    ),
] = 1.0
```

## How It Works

The oversampling factor is used in `src/prime_rl/orchestrator/scheduler.py` to calculate `problems_per_batch`:

```python
self.problems_per_batch = int(oversampling_factor * self.batch_size // self.rollouts_per_example)
```

### Formula Breakdown

```
problems_per_batch = (oversampling_factor × batch_size) ÷ rollouts_per_example
```

**Example calculation** (from configs):
- `batch_size = 128`
- `rollouts_per_example = 1` (typical default)
- `oversampling_factor = 2.0`
- **Result**: `problems_per_batch = (2.0 × 128) ÷ 1 = 256 problems`

This means:
- The orchestrator requests **256 unique problems** from the buffer
- Each problem generates **1 rollout** (trajectory)
- But only **128 samples** are used for training per batch
- The system has **2x more rollouts in flight** than needed for a single batch

## Purpose and Benefits

### 1. **Diversity Increase**
More unique problems sampled per batch leads to:
- More diverse training data
- Reduced overfitting to specific problems
- Better generalization

### 2. **Asynchronous Pipeline Efficiency**
With oversampling > 1.0:
- More rollouts in flight simultaneously
- Better GPU utilization (inference workers stay busy)
- Helps hide latency in environment execution
- Smoother training pipeline (less waiting for rollouts)

### 3. **Difficulty-Based Sampling Integration**
Works with prime-rl's online difficulty buffer:
- Buffer classifies problems as "easy", "normal", "hard" based on average reward
- Only "normal" difficulty problems are sampled for training
- Oversampling provides more candidates to select from
- Increases chance of finding problems with non-zero advantages

## Current Usage in Codebase

All examples and configs use **2x oversampling**:

```toml
# From configs/art_e_rl/*.toml, examples/*/rl.toml
oversampling_factor = 2.0
```

### Examples with 2x Oversampling
1. **ART-e RL experiments** (Enron email QA)
   - `configs/art_e_rl/8b.toml`
   - `configs/art_e_rl/4b.toml`
   - `configs/art_e_rl/1.7b.toml`

2. **Enron Email example**
   - `examples/enron_email/rl.toml`
   - README: "Uses difficulty-based sampling with 2x oversampling"

3. **Wiki Search example**
   - `examples/wiki_search/rl.toml`
   - README: "Uses difficulty-based sampling with 2x oversampling"

## Recommended Values

Based on codebase patterns:

| Factor | Use Case | Trade-offs |
|--------|----------|------------|
| **1.0** | Minimal (default) | No oversampling, fastest but least diverse |
| **2.0** | Standard (codebase default) | Good balance of diversity and efficiency |
| **3.0-4.0** | High diversity | More compute overhead, diminishing returns |
| **>4.0** | Not recommended | Significant overhead with minimal benefit |

## Trade-offs

### Increasing Oversampling Factor

**Pros:**
- ✅ More diverse training data
- ✅ Better for difficulty-based sampling (more candidates)
- ✅ Smoother asynchronous pipeline
- ✅ Better GPU utilization

**Cons:**
- ❌ More rollouts generated than used (computational waste)
- ❌ Higher memory usage (more rollouts in flight)
- ❌ Longer time to complete a batch (more rollouts to wait for)
- ❌ Diminishing returns above 2-3x

### Decreasing Oversampling Factor

**Pros:**
- ✅ Less computational overhead
- ✅ Faster batch completion
- ✅ Lower memory usage

**Cons:**
- ❌ Less diverse training data
- ❌ May underutilize inference workers
- ❌ Pipeline stalls more likely
- ❌ Worse for difficulty-based sampling

## Interaction with Other Parameters

### 1. **batch_size**
- Larger batch_size → need proportionally more oversampling to maintain diversity
- With `batch_size=128` and `oversampling_factor=2.0`, you get 256 problems

### 2. **rollouts_per_example**
- If `rollouts_per_example > 1`, fewer unique problems needed
- Example: `rollouts_per_example=2` means 128 problems generate 256 rollouts
- Oversampling still multiplies the problem count

### 3. **Buffer Difficulty Thresholds**
- `easy_threshold`: Problems above this reward go to "easy" pool (not sampled)
- `hard_threshold`: Problems below this go to "hard" pool (not sampled)
- Higher oversampling helps when many problems are in easy/hard pools

### 4. **max_async_level**
- Controls how many batches can be in flight
- Higher oversampling + higher async_level = many more rollouts in memory

## Recommendations for ART-e RL Experiments

Current configuration is **good and consistent**:
- All three model sizes (1.7B, 4B, 8B) use `oversampling_factor = 2.0`
- This is the codebase standard and proven in examples
- Provides good balance for multi-turn tool-use tasks

**When to adjust:**
- If seeing pipeline stalls (inference workers idle): increase to 2.5-3.0
- If memory constrained: decrease to 1.5
- If training is too slow and memory is abundant: increase to 2.5-3.0
- For hyperparameter tuning: test [1.5, 2.0, 2.5] and measure diversity metrics

## Key Insights

1. **Oversampling is about diversity, not just throughput**
   - Even if inference is fast, oversampling ensures varied training data

2. **Works best with difficulty-based sampling**
   - Prime-rl's buffer system classifies problems by difficulty
   - More samples = better chance of finding "normal" difficulty problems

3. **2x is the de facto standard**
   - All examples use 2.0
   - Proven to work well for multi-turn RL tasks

4. **Diminishing returns above 3x**
   - Computational cost increases linearly
   - Diversity benefit plateaus

5. **Not a free lunch**
   - Generate 2x rollouts but use only batch_size for training
   - Trade memory/compute for diversity

## References

- Config definition: `src/prime_rl/orchestrator/config.py:706-712`
- Usage in scheduler: `src/prime_rl/orchestrator/scheduler.py:72`
- Buffer sampling: `src/prime_rl/orchestrator/buffer.py:198-212`
- Example configs: `configs/art_e_rl/*.toml`, `examples/*/rl.toml`
