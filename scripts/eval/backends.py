"""Model backends for ART-e evaluation.

Provides OpenAI, Anthropic, and vLLM local backends with a unified response format.
"""

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import Union

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from scripts.eval.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class BackendResponse:
    """Unified response format (OAI-style tool_calls)."""

    content: str | None
    tool_calls: list[dict] | None
    stop_reason: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAIBackend:
    """Backend for OpenAI API models (GPT-5, GPT-5-mini, etc.)."""

    def __init__(self, config: ModelConfig):
        api_key = os.environ.get(config.api_key_env, "EMPTY")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=config.base_url,
        )
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> BackendResponse:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            max_completion_tokens=self.max_tokens,
        )
        choice = response.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        prompt_tokens = 0
        completion_tokens = 0
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0

        return BackendResponse(
            content=msg.content,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class AnthropicBackend:
    """Backend for Anthropic API with tool use."""

    def __init__(self, config: ModelConfig):
        api_key = os.environ.get(config.api_key_env, "")
        self.client = AsyncAnthropic(api_key=api_key)
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> BackendResponse:
        system_msg, anthropic_messages = _convert_messages_to_anthropic(messages)
        anthropic_tools = _oai_tools_to_anthropic(tools)

        response = await self.client.messages.create(
            model=self.model_name,
            system=system_msg,
            messages=anthropic_messages,
            tools=anthropic_tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content_text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )

        prompt_tokens = 0
        completion_tokens = 0
        if response.usage:
            prompt_tokens = response.usage.input_tokens or 0
            completion_tokens = response.usage.output_tokens or 0

        return BackendResponse(
            content=content_text or None,
            tool_calls=tool_calls or None,
            stop_reason=response.stop_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class VLLMLocalBackend:
    """Backend for vLLM local inference with LoRA adapter support.

    Loads the model once into GPU memory and runs inference locally.
    Tool calls are parsed from hermes-format text output.
    """

    def __init__(self, config: ModelConfig):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        logger.info(f"Loading vLLM model: {config.model_name}")
        llm_kwargs: dict = {
            "model": config.model_name,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
            "trust_remote_code": True,
        }
        if config.adapter_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = config.lora_rank or 32

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.lora_request = None
        if config.adapter_path:
            self.lora_request = LoRARequest(
                "art-e-rl", 1, config.adapter_path
            )
        logger.info("vLLM model loaded successfully")

    async def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> BackendResponse:
        result = await asyncio.to_thread(
            self._sync_chat, messages, tools
        )
        return result

    def _sync_chat(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> BackendResponse:
        outputs = self.llm.chat(
            messages=messages,
            tools=tools,
            sampling_params=self.sampling_params,
            lora_request=self.lora_request,
        )
        output = outputs[0]
        completion = output.outputs[0]
        text = completion.text
        finish_reason = completion.finish_reason

        # Token counts: prompt from input, completion from generated token_ids
        prompt_tokens = len(output.prompt_token_ids) if output.prompt_token_ids else 0
        completion_tokens = len(completion.token_ids) if completion.token_ids else 0

        # Parse hermes-format tool calls from generated text
        tool_calls = _parse_hermes_tool_calls(text)

        if tool_calls:
            content = _strip_hermes_tags(text)
            return BackendResponse(
                content=content or None,
                tool_calls=tool_calls,
                stop_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        return BackendResponse(
            content=text or None,
            tool_calls=None,
            stop_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


# --- Backend factory ---

BackendType = Union[OpenAIBackend, AnthropicBackend, VLLMLocalBackend]


def create_backend(config: ModelConfig) -> BackendType:
    """Factory to create the appropriate backend."""
    if config.backend == "openai":
        return OpenAIBackend(config)
    elif config.backend == "anthropic":
        return AnthropicBackend(config)
    elif config.backend == "vllm_local":
        return VLLMLocalBackend(config)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")


# --- Hermes tool call parsing ---

_HERMES_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)


def _parse_hermes_tool_calls(text: str) -> list[dict] | None:
    """Parse hermes-format tool calls from generated text.

    Hermes format:
        <tool_call>
        {"name": "func_name", "arguments": {"key": "value"}}
        </tool_call>
    """
    matches = _HERMES_TOOL_CALL_RE.findall(text)
    if not matches:
        return None

    tool_calls = []
    for match in matches:
        try:
            parsed = json.loads(match)
            name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            )
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse hermes tool call: {match[:100]}")
            continue

    return tool_calls or None


def _strip_hermes_tags(text: str) -> str:
    """Remove hermes tool call blocks from text, leaving other content."""
    return _HERMES_TOOL_CALL_RE.sub("", text).strip()


# --- Anthropic format conversion helpers ---


def _oai_tools_to_anthropic(oai_tools: list[dict]) -> list[dict]:
    """Convert OAI tool format to Anthropic tool format."""
    result = []
    for tool in oai_tools:
        func = tool["function"]
        schema = dict(func["parameters"])
        schema.pop("strict", None)
        result.append(
            {
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": schema,
            }
        )
    return result


def _convert_messages_to_anthropic(
    messages: list[dict],
) -> tuple[str, list[dict]]:
    """Convert OAI message format to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    """
    system_prompt = ""
    anthropic_messages: list[dict] = []

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg["role"]

        if role == "system":
            system_prompt = msg["content"]
            i += 1

        elif role == "user":
            anthropic_messages.append({"role": "user", "content": msg["content"]})
            i += 1

        elif role == "assistant":
            content_blocks: list[dict] = []
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg["content"]})
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc["function"]
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": func["name"],
                            "input": json.loads(func["arguments"]),
                        }
                    )
            anthropic_messages.append({"role": "assistant", "content": content_blocks})
            i += 1

        elif role == "tool":
            # Collect consecutive tool results into a single user message
            tool_results: list[dict] = []
            while i < len(messages) and messages[i]["role"] == "tool":
                tool_msg = messages[i]
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_msg["tool_call_id"],
                        "content": tool_msg["content"],
                    }
                )
                i += 1
            anthropic_messages.append({"role": "user", "content": tool_results})

        else:
            i += 1

    return system_prompt, anthropic_messages
