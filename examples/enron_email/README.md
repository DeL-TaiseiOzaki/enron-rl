# Enron Email Search

In this example, we demonstrate how to train `Qwen3-8B` to answer questions by searching through the Enron email corpus using multi-turn tool use. This example highlights several key features of PRIME-RL and verifiers environment features:

- **Single-file configuration**: All training settings (trainer, orchestrator, and inference) are specified in a single `rl.toml` file
- **LoRA training**: Efficient fine-tuning using LoRA (Low-Rank Adaptation) on attention and MLP layers
- **Multi-turn tool use**: The model learns to use email search and retrieval tools across multiple turns via `ToolEnv` via native function calling
- **LLM judges**: Uses an LLM judge to evaluate answer correctness against reference answers
- **Online difficulty buffer**: Uses difficulty-based sampling to ensure rollouts have strictly non-zero advantages

> This example runs on 4 GPUs (1 for inference, 3 for training).

## Setup

Install the environment:

```bash
prime env install art-e
```

Verify installation:

```bash
uv run python -c "import art_e"
```

Create the email database (required on first run):

```bash
uv run python -c "from art_e.utils.local_email_db import generate_database; generate_database()"
```

Set up your OpenAI API key for the judge model:

```bash
export OPENAI_API_KEY=your_api_key_here
```

Start the tmux session:

```bash
bash scripts/tmux.sh
```

## Task

The art-e environment requires the model to answer questions about emails by:

1. **Searching** the inbox using keyword-based full-text search (FTS5)
2. **Reading** full email content by message ID
3. **Answering** the question by calling `return_final_answer` with the answer and source message IDs

The environment provides three tools:
- `search_inbox(inbox, keywords, sent_before)`: Search emails by keywords and date filter, returning up to 10 matching snippets
- `read_email(message_id)`: Retrieve the full email content (subject, body, recipients, date)
- `return_final_answer(answer, sources)`: Submit the final answer and evidence. The inference loop stops when this is called

The dataset contains ~44,000 training questions and ~1,700 evaluation questions generated from the [Enron email dataset](https://huggingface.co/datasets/corbt/enron-emails). See the [ART-E blog post](https://openpipe.ai/blog/art-e-mail-agent) for more details.

## Scoring

The environment uses a composite rubric combining:

1. **JudgeRubric** (weight 1.0): An LLM judge (default: `gpt-4.1-mini`) compares the model's answer against the reference answer. Returns 1.0 for correct, -1.0 for hallucinated, and 0.0 for "I don't know" or empty answers
2. **ToolCountRubric** (weight 0.1): Rewards efficient tool usage with `1 - (tool_calls / max_turns)`. Returns 0.0 if more than `max_turns` tool calls are made

| Metric | Meaning |
| ------ | ------- |
| `judge_reward` | 1.0 correct, 0.0 "I don't know", -1.0 hallucinated |
| `tool_count_reward` | `1 - (num_tool_calls / max_turns)`, 0.0 if over limit |

## Configuration

This example uses a **single `rl.toml` file** that contains all configuration for trainer, orchestrator, and inference in a single place. This simplifies configuration for single-node training via `rl.py`.

Key configuration highlights:

- **LoRA training**: Rank 16 for efficient fine-tuning
- **Tool calling**: Uses Hermes parser for automatic tool selection with Qwen3-8B
- **Multi-turn**: Up to 10 turns per episode (configurable via environment args)
- **Online difficulty buffer**: Uses difficulty-based sampling with 2x oversampling

## Baseline Evaluation

Start the inference server:

```bash
# In the `Inference` pane
uv run inference --enable-lora --model.name Qwen/Qwen3-8B --model.enable_auto_tool_choice --model.tool_call_parser hermes
```

Evaluate the base model:

```bash
# In the `Trainer` pane
uv run vf-eval art_e \
  -m Qwen/Qwen3-8B \
  -b http://localhost:8000/v1 \
  -n 20 \
  --max-tokens 2048 \
  --env-args '{"max_turns": 10, "judge_model": "gpt-4.1-mini", "use_tool_count_reward": true}'
```

## RL Training

Train with the unified config file:

```bash
# In the `Trainer` pane
uv run rl @ examples/enron_email/rl.toml \
  --wandb.project your-project-name \
  --wandb.name your-run-name
```

The unified config file automatically configures:
- **Trainer**: LoRA fine-tuning with specified hyperparameters
- **Orchestrator**: Rollout generation with tool calling enabled
- **Inference**: vLLM server for Qwen3-8B with tool parsing enabled

This will write weight checkpoints in `outputs/weights/step_*`. Upload the final checkpoint to HuggingFace:

```bash
uv run hf upload <user>/Qwen3-8B-EnronEmail-RL outputs/weights/step_1000
```

## Evaluation

Evaluate your trained model:

```bash
# In the `Inference` pane
uv run inference --enable-lora --model.name <user>/Qwen3-8B-EnronEmail-RL --model.enable_auto_tool_choice --model.tool_call_parser hermes
```

```bash
# In the `Trainer` pane
uv run vf-eval art_e \
  -m <user>/Qwen3-8B-EnronEmail-RL \
  -b http://localhost:8000/v1 \
  -n 100 \
  --max-tokens 2048 \
  --env-args '{"max_turns": 10, "judge_model": "gpt-4.1-mini", "use_tool_count_reward": true}'
```

## Environment Arguments

The art-e environment supports several configuration options:

| Argument | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_turns` | int | `10` | Maximum number of tool-use turns per episode |
| `use_tool_count_reward` | bool | `true` | Whether to include tool count reward in the rubric |
| `judge_model` | str | `"gpt-4.1-mini"` | Judge model name |
| `judge_client` | OpenAI | `OpenAI()` | OpenAI client instance for the judge |
| `judge_prompt` | str | `DEFAULT_JUDGE_PROMPT` | Custom prompt for the judge |

You can pass these via the `--env-args` flag in `vf-eval` or configure them in your `rl.toml`:

```toml
[[orchestrator.env]]
id = "art_e"
args = { max_turns = 5, judge_model = "gpt-4.1" }
```

## Notes

- The first run requires building the email database from the HuggingFace dataset (`corbt/enron-emails`)
- Ensure `OPENAI_API_KEY` is set in your environment for the judge model
- The SQLite database is stored locally and persists across runs
- Tool calling requires `enable_auto_tool_choice = true` and a compatible parser (Hermes is recommended)
