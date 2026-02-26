"""CLI entry point for ART-e model evaluation.

Usage:
    uv run python scripts/eval/run_eval.py --config configs/eval/art_e_comparison.yaml
    uv run python scripts/eval/run_eval.py --config configs/eval/art_e_comparison.yaml --models gpt-5-mini
    uv run python scripts/eval/run_eval.py --config configs/eval/art_e_comparison.yaml --resume
"""

import argparse
import asyncio
import json
import logging
import os
import random
from pathlib import Path

from dotenv import load_dotenv

from datasets import load_from_disk
from openai import OpenAI
from tqdm.asyncio import tqdm

import art_e.utils.local_email_db as email_db
import art_e.utils.search_tools as search_tools
from scripts.eval.backends import create_backend
from scripts.eval.config import EvalConfig, ModelConfig, load_config
from scripts.eval.runner import (
    AgentResult,
    build_system_prompt,
    compute_metrics,
    compute_tool_count_reward,
    get_oai_tools,
    judge_answer,
    run_agent_loop,
)

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def load_and_sample_dataset(config: EvalConfig) -> list[dict]:
    """Load test dataset and randomly sample examples."""
    ds = load_from_disk(config.dataset.path)
    test_ds = ds[config.dataset.split]

    # Format dataset (same as art_e.format_dataset but inline)
    max_turns = config.experiment.max_turns
    examples = []
    for idx, row in enumerate(test_ds):
        system_prompt = build_system_prompt(
            inbox_address=row["inbox_address"],
            query_date=row["query_date"],
            max_turns=max_turns,
        )
        example = {
            "example_id": row.get("id", idx),
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["question"]},
            ],
            "info": {
                "answer": row["answer"],
                "message_ids": row["message_ids"],
                "inbox_address": row["inbox_address"],
                "query_date": row["query_date"],
            },
        }
        examples.append(example)

    # Sample if needed
    num_samples = min(config.experiment.num_samples, len(examples))
    if num_samples < len(examples):
        rng = random.Random(config.experiment.seed)
        examples = rng.sample(examples, num_samples)
        logger.info(f"Sampled {num_samples} examples from {len(test_ds)} (seed={config.experiment.seed})")
    else:
        logger.info(f"Using all {len(examples)} examples")

    return examples


def init_db(db_path: str) -> None:
    """Initialize the email database connection for search_tools."""
    abs_path = str(Path(db_path).resolve())
    email_db.DEFAULT_DB_PATH = abs_path
    # Reset cached connection so it picks up the new path
    search_tools.conn = None
    logger.info(f"Database path set to: {abs_path}")


def load_existing_results(results_path: Path) -> set[int]:
    """Load already-evaluated example_ids from a results file for resume."""
    if not results_path.exists():
        return set()
    ids = set()
    with open(results_path) as f:
        for line in f:
            try:
                result = json.loads(line)
                ids.add(result["example_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def save_result(result: AgentResult, path: Path) -> None:
    """Append a single result to a JSONL file."""
    row = {
        "example_id": result.example_id,
        "question": result.question,
        "reference_answer": result.reference_answer,
        "generated_answer": result.generated_answer,
        "sources": result.sources,
        "judge_score": result.judge_score,
        "tool_count_reward": result.tool_count_reward,
        "total_reward": result.total_reward,
        "tool_call_count": result.tool_call_count,
        "turns_used": result.turns_used,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "error": result.error,
        "messages": result.messages,
    }
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def evaluate_model(
    model_config: ModelConfig,
    examples: list[dict],
    oai_tools: list[dict],
    max_turns: int,
    judge_client: OpenAI,
    judge_model: str,
    output_dir: Path,
    resume: bool = False,
    reward_correct: float = 1.0,
    reward_wrong: float = -1.0,
    reward_idk: float = 0.0,
    use_tool_count_reward: bool = True,
) -> list[AgentResult]:
    """Evaluate a single model on all examples."""
    model_dir = output_dir / model_config.name
    model_dir.mkdir(parents=True, exist_ok=True)
    results_path = model_dir / "results.jsonl"

    # Resume: skip already-evaluated examples
    eval_examples = examples
    if resume:
        done_ids = load_existing_results(results_path)
        eval_examples = [ex for ex in examples if ex["example_id"] not in done_ids]
        logger.info(
            f"[{model_config.name}] Resume: {len(done_ids)} done, {len(eval_examples)} remaining"
        )
        if not eval_examples:
            logger.info(f"[{model_config.name}] All examples already evaluated")
            return []

    backend = create_backend(model_config)
    semaphore = asyncio.Semaphore(model_config.max_concurrent)
    results: list[AgentResult] = []

    tool_count_weight = 0.05 if use_tool_count_reward else 0.0

    async def eval_one(example: dict) -> AgentResult:
        async with semaphore:
            result = await run_agent_loop(
                backend=backend,
                example=example,
                oai_tools=oai_tools,
                max_turns=max_turns,
            )

            # Score with judge
            try:
                result.judge_score = await asyncio.to_thread(
                    _sync_judge,
                    result.question,
                    result.reference_answer,
                    result.generated_answer,
                    judge_client,
                    judge_model,
                    reward_correct,
                    reward_wrong,
                    reward_idk,
                )
            except Exception as e:
                logger.error(f"Judge error for example {result.example_id}: {e}")
                result.judge_score = 0.0

            # Compute tool count reward and total
            # Only award efficiency bonus when agent actually produced an answer
            if result.error is not None or result.generated_answer is None:
                result.tool_count_reward = 0.0
            else:
                result.tool_count_reward = compute_tool_count_reward(
                    result.tool_call_count, max_turns
                )
            result.total_reward = (
                1.0 * result.judge_score + tool_count_weight * result.tool_count_reward
            )

            # Save immediately
            save_result(result, results_path)
            return result

    pbar = tqdm(total=len(eval_examples), desc=f"[{model_config.name}]")

    async def eval_and_track(example: dict) -> AgentResult:
        result = await eval_one(example)
        pbar.update(1)

        # Update running stats
        results.append(result)
        scored = [r for r in results if r.error is None]
        if scored:
            avg = sum(r.judge_score for r in scored) / len(scored)
            pbar.set_postfix({"avg_score": f"{avg:.3f}"})

        return result

    await asyncio.gather(*[eval_and_track(ex) for ex in eval_examples])
    pbar.close()

    return results


def _sync_judge(
    question: str,
    reference_answer: str,
    generated_answer: str | None,
    judge_client: OpenAI,
    judge_model: str,
    reward_correct: float = 1.0,
    reward_wrong: float = -1.0,
    reward_idk: float = 0.0,
) -> float:
    """Synchronous wrapper for judge_answer (used in asyncio.to_thread)."""
    if generated_answer is None or generated_answer.strip() == "" or generated_answer == "I don't know":
        return reward_idk

    from scripts.eval.runner import EVAL_JUDGE_PROMPT, JudgeOutput

    user_msg = (
        f"Question: {question}\n"
        f"Reference Answer: {reference_answer}\n"
        f"AI Answer: {generated_answer}"
    )

    response = judge_client.chat.completions.parse(
        model=judge_model,
        messages=[
            {"role": "system", "content": EVAL_JUDGE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=JudgeOutput,
    )

    parsed = response.choices[0].message.parsed
    return reward_correct if parsed.accept else reward_wrong


def save_summary(output_dir: Path, all_metrics: dict) -> None:
    """Save aggregated metrics summary."""
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")


def print_comparison_table(all_metrics: dict) -> None:
    """Print a comparison table of all models."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    models = all_metrics.get("models", {})
    if not models:
        print("No results to display.")
        return

    # Header
    header = (
        f"{'Model':<22} {'Acc':>6} {'Score':>7} {'Halluc':>7} {'IDK':>6} "
        f"{'Tools':>6} {'Tokens':>8} {'Tok/Corr':>9} {'Time':>7}"
    )
    print(header)
    print("-" * 90)

    for name, metrics in models.items():
        tok_per_corr = metrics.get("tokens_per_correct_answer", float("inf"))
        tok_per_corr_str = f"{tok_per_corr:>7.0f}" if tok_per_corr != float("inf") else "    inf"
        row = (
            f"{name:<22} "
            f"{metrics.get('accuracy', 0):>5.1%} "
            f"{metrics.get('avg_judge_score', 0):>+6.3f} "
            f"{metrics.get('hallucination_rate', 0):>6.1%} "
            f"{metrics.get('idk_rate', 0):>5.1%} "
            f"{metrics.get('avg_tool_calls', 0):>5.1f} "
            f"{metrics.get('avg_total_tokens', 0):>7.0f} "
            f"{tok_per_corr_str} "
            f"{metrics.get('avg_elapsed_seconds', 0):>6.1f}s"
        )
        print(row)

    print("=" * 90)


async def main() -> None:
    parser = argparse.ArgumentParser(description="ART-e Model Evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--models", nargs="*", help="Evaluate only these models (by name)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    parser.add_argument("--num-samples", type=int, help="Override num_samples from config")
    parser.add_argument("--output-dir", type=str, help="Override output_dir from config")
    args = parser.parse_args()

    setup_logging()
    load_dotenv()

    # Load config
    config = load_config(args.config)
    if args.num_samples is not None:
        config.experiment.num_samples = args.num_samples
    if args.output_dir is not None:
        config.experiment.output_dir = args.output_dir

    # Filter models if specified
    if args.models:
        config.models = [m for m in config.models if m.name in args.models]
        if not config.models:
            logger.error(f"No models matched: {args.models}")
            return

    # Initialize
    init_db(config.dataset.db_path)
    examples = load_and_sample_dataset(config)
    oai_tools = get_oai_tools()
    output_dir = Path(config.experiment.output_dir) / config.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize judge client
    judge_api_key = os.environ.get(config.judge.api_key_env, "")
    judge_client = OpenAI(api_key=judge_api_key)

    logger.info(f"Evaluating {len(config.models)} model(s) on {len(examples)} examples")
    logger.info(f"Output directory: {output_dir}")

    # Evaluate each model sequentially
    all_metrics: dict = {
        "experiment": config.experiment.name,
        "num_samples": len(examples),
        "seed": config.experiment.seed,
        "models": {},
    }

    for model_config in config.models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_config.name} ({model_config.backend}: {model_config.model_name})")
        logger.info(f"{'='*60}")

        results = await evaluate_model(
            model_config=model_config,
            examples=examples,
            oai_tools=oai_tools,
            max_turns=config.experiment.max_turns,
            judge_client=judge_client,
            judge_model=config.judge.model,
            output_dir=output_dir,
            resume=args.resume,
            reward_correct=config.judge.reward_correct,
            reward_wrong=config.judge.reward_wrong,
            reward_idk=config.judge.reward_idk,
        )

        if results:
            metrics = compute_metrics(results)
            all_metrics["models"][model_config.name] = metrics
            logger.info(
                f"[{model_config.name}] accuracy={metrics['accuracy']:.1%}, "
                f"avg_score={metrics['avg_judge_score']:+.3f}, "
                f"hallucination={metrics['hallucination_rate']:.1%}, "
                f"avg_tokens={metrics['avg_total_tokens']:.0f}, "
                f"tokens/correct={metrics['tokens_per_correct_answer']:.0f}"
            )

    # Save summary and print comparison
    save_summary(output_dir, all_metrics)
    print_comparison_table(all_metrics)


if __name__ == "__main__":
    asyncio.run(main())
