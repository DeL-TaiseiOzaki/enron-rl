"""Agent loop, judge scoring, and metrics for ART-e evaluation."""

import json
import logging
import time
from dataclasses import dataclass, field

from openai import OpenAI
from pydantic import BaseModel

from art_e.art_e import DEFAULT_JUDGE_PROMPT, get_oai_tool_json_schema, return_final_answer
from art_e.utils.search_tools import read_email, search_inbox

from scripts.eval.backends import BackendType

logger = logging.getLogger(__name__)

# Extended judge prompt for evaluation: adds abbreviation/expansion equivalence
EVAL_JUDGE_PROMPT = DEFAULT_JUDGE_PROMPT.rstrip() + """

Additional rule:
5. Abbreviations and their expanded forms are semantically equivalent and must be treated as identical. For example: "COO" = "Chief Operating Officer", "MD" = "Managing Director", "VP" = "Vice President", "CEO" = "Chief Executive Officer", "CFO" = "Chief Financial Officer", "CTO" = "Chief Technology Officer", "EVP" = "Executive Vice President", "SVP" = "Senior Vice President". If the only difference between the Reference answer and the AI answer is an abbreviation vs. its expansion (or vice versa), *accept* must be **true**."""

# Tool functions for schema generation and execution
TOOLS = [search_inbox, read_email, return_final_answer]
TOOL_DISPATCH = {
    "search_inbox": search_inbox,
    "read_email": read_email,
}


class JudgeOutput(BaseModel):
    thinking: str
    accept: bool


@dataclass
class AgentResult:
    """Result of a single agent evaluation run."""

    example_id: int
    question: str
    reference_answer: str
    generated_answer: str | None = None
    sources: list[str] | None = None
    judge_score: float = 0.0
    tool_count_reward: float = 0.0
    total_reward: float = 0.0
    tool_call_count: int = 0
    turns_used: int = 0
    elapsed_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str | None = None
    messages: list[dict] = field(default_factory=list)


def build_system_prompt(inbox_address: str, query_date: str, max_turns: int) -> str:
    """Build the system prompt matching art_e's format_dataset()."""
    return (
        f"You are an email search agent. You are given a user query and a list of "
        f"tools you can use to search the user's email. Use the tools to search the "
        f"user's emails and find the answer to the user's query. You may take up to "
        f"{max_turns} turns to find the answer, so if your first seach doesn't find "
        f"the answer, you can try with different keywords.\n\n"
        f"        To respond to the user's query, you should call the "
        f"`return_final_answer` function with the answer and any sources used to "
        f"find the answer.\n\n"
        f"User's email address is {inbox_address}\n"
        f"Today's date is {query_date}"
    )


def get_oai_tools() -> list[dict]:
    """Get OpenAI-format tool definitions for the art_e tools."""
    return get_oai_tool_json_schema(TOOLS)


async def run_agent_loop(
    backend: BackendType,
    example: dict,
    oai_tools: list[dict],
    max_turns: int,
) -> AgentResult:
    """Execute multi-turn agent loop for a single example.

    The agent iteratively calls tools (search_inbox, read_email) and
    submits a final answer via return_final_answer.
    """
    start_time = time.perf_counter()
    info = example["info"]
    prompt_messages = example["prompt"]

    # Start from the formatted prompt (system + user messages)
    messages = list(prompt_messages)
    question = next(
        (m["content"] for m in prompt_messages if m["role"] == "user"), ""
    )

    tool_call_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for turn in range(max_turns):
        try:
            response = await backend.chat_completion(
                messages=messages,
                tools=oai_tools,
            )
        except Exception as e:
            logger.error(f"Backend error at turn {turn}: {e}")
            elapsed = time.perf_counter() - start_time
            return AgentResult(
                example_id=example.get("example_id", example.get("id", 0)),
                question=question,
                reference_answer=info["answer"],
                turns_used=turn,
                elapsed_seconds=elapsed,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
                error=str(e),
                messages=messages,
            )

        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens

        # Build assistant message in OAI format
        assistant_msg: dict = {"role": "assistant", "content": response.content}
        if response.tool_calls:
            assistant_msg["tool_calls"] = response.tool_calls
        messages.append(assistant_msg)

        # No tool calls — agent finished without submitting answer
        if not response.tool_calls:
            break

        # Process each tool call
        for tc in response.tool_calls:
            func_name = tc["function"]["name"]
            try:
                func_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                func_args = {}

            if func_name == "return_final_answer":
                # Agent submitted answer — stop loop
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "Answer submitted.",
                    }
                )
                elapsed = time.perf_counter() - start_time
                tool_call_count += len(response.tool_calls)
                return AgentResult(
                    example_id=example.get("example_id", example.get("id", 0)),
                    question=question,
                    reference_answer=info["answer"],
                    generated_answer=func_args.get("answer"),
                    sources=func_args.get("sources"),
                    tool_call_count=tool_call_count,
                    turns_used=turn + 1,
                    elapsed_seconds=elapsed,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                    messages=messages,
                )

            # Execute search/read tool
            tool_func = TOOL_DISPATCH.get(func_name)
            if tool_func:
                try:
                    result = tool_func(**func_args)
                except Exception as e:
                    result = f"Tool error: {e}"
            else:
                result = f"Unknown tool: {func_name}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result),
                }
            )

        tool_call_count += len(response.tool_calls)

    # Ran out of turns without submitting answer
    elapsed = time.perf_counter() - start_time
    return AgentResult(
        example_id=example.get("example_id", example.get("id", 0)),
        question=question,
        reference_answer=info["answer"],
        tool_call_count=tool_call_count,
        turns_used=max_turns,
        elapsed_seconds=elapsed,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
        messages=messages,
    )


async def judge_answer(
    question: str,
    reference_answer: str,
    generated_answer: str | None,
    judge_client: OpenAI,
    judge_model: str,
    judge_prompt: str = DEFAULT_JUDGE_PROMPT,
    reward_correct: float = 1.0,
    reward_wrong: float = -1.0,
    reward_idk: float = 0.0,
) -> float:
    """Score a generated answer using the LLM judge.

    Returns reward_correct (correct), reward_wrong (wrong), reward_idk (IDK/timeout/empty).
    Matches art_e.judge_reward logic exactly.
    """
    if generated_answer is None or generated_answer.strip() == "" or generated_answer == "I don't know":
        return reward_idk

    user_msg = (
        f"Question: {question}\n"
        f"Reference Answer: {reference_answer}\n"
        f"AI Answer: {generated_answer}"
    )

    response = judge_client.chat.completions.parse(
        model=judge_model,
        messages=[
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format=JudgeOutput,
    )

    parsed = response.choices[0].message.parsed
    return reward_correct if parsed.accept else reward_wrong


def compute_tool_count_reward(tool_call_count: int, max_turns: int) -> float:
    """Compute efficiency reward based on tool call count.

    Matches art_e.tool_count_reward logic.
    """
    if tool_call_count > max_turns:
        return 0.0
    return 1 - tool_call_count / max_turns


def compute_metrics(results: list[AgentResult]) -> dict:
    """Compute aggregate metrics from evaluation results."""
    total = len(results)
    if total == 0:
        return {}

    correct = sum(1 for r in results if r.judge_score > 0)
    wrong = sum(1 for r in results if r.judge_score < 0)
    idk = sum(1 for r in results if r.judge_score == 0.0)
    errors = sum(1 for r in results if r.error is not None)

    judge_scores = [r.judge_score for r in results]
    tool_counts = [r.tool_call_count for r in results]
    turns = [r.turns_used for r in results]
    times = [r.elapsed_seconds for r in results]
    prompt_tokens = [r.prompt_tokens for r in results]
    completion_tokens = [r.completion_tokens for r in results]
    total_tokens = [r.total_tokens for r in results]

    # Token efficiency: tokens per correct answer
    correct_results = [r for r in results if r.judge_score > 0]
    tokens_per_correct = (
        sum(r.total_tokens for r in correct_results) / len(correct_results)
        if correct_results
        else float("inf")
    )

    return {
        "total": total,
        "accuracy": correct / total,
        "avg_judge_score": sum(judge_scores) / total,
        "hallucination_rate": wrong / total,
        "idk_rate": idk / total,
        "error_rate": errors / total,
        "avg_tool_calls": sum(tool_counts) / total,
        "avg_turns": sum(turns) / total,
        "avg_elapsed_seconds": sum(times) / total,
        "avg_prompt_tokens": sum(prompt_tokens) / total,
        "avg_completion_tokens": sum(completion_tokens) / total,
        "avg_total_tokens": sum(total_tokens) / total,
        "total_tokens_all": sum(total_tokens),
        "tokens_per_correct_answer": tokens_per_correct,
    }
