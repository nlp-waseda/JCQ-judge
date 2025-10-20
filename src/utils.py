import asyncio
import json
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

OPENAI_MAX_N_PER_REQUEST = 8


def load_records(file: str | Path) -> list[dict[str, Any]]:
    with open(file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return records


def save_answers(
    path: Path, questions: list[dict[str, Any]], answers: list[list[str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        answer_id = 1
        for record, answer_choices in zip(questions, answers):
            for answer in answer_choices:
                result = {
                    "id": answer_id,
                    "question_id": record["id"],
                    "answer": answer,
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                answer_id += 1


class OpenAIGenerator:
    def __init__(self, concurrency: int):
        self._concurrency = concurrency
        self._client: AsyncOpenAI | None = None
        self._semaphore: asyncio.Semaphore | None = None

    async def __aenter__(self):
        self._client = AsyncOpenAI()
        self._semaphore = asyncio.Semaphore(self._concurrency)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client:
            await self._client.close()

    async def _generate(self, **params: Any) -> list[str]:
        if not self._client or not self._semaphore:
            raise RuntimeError("Generator not initialized")

        async with self._semaphore:
            completion = await self._client.chat.completions.create(**params)
            return [choice.message.content for choice in completion.choices]

    async def generate_concurrently(
        self, prompts: list[list[dict[str, str]]], **params: Any
    ) -> list[list[str]]:
        params = {k: v for k, v in params.items() if v is not None}
        n = params.get("n", 1)

        if n <= OPENAI_MAX_N_PER_REQUEST:
            tasks = [
                self._generate(messages=messages, **params) for messages in prompts
            ]
            return await tqdm_asyncio.gather(*tasks)

        tasks = []
        for messages in prompts:
            remaining = n
            prompt_tasks = []
            while remaining > 0:
                current_n = min(OPENAI_MAX_N_PER_REQUEST, remaining)
                batch_params = {**params, "n": current_n}
                prompt_tasks.append(self._generate(messages=messages, **batch_params))
                remaining -= current_n
            tasks.append(prompt_tasks)

        all_tasks = [task for prompt_tasks in tasks for task in prompt_tasks]
        all_results = await tqdm_asyncio.gather(*all_tasks)

        results = []
        idx = 0
        for prompt_tasks in tasks:
            num_batches = len(prompt_tasks)
            batch_results = all_results[idx : idx + num_batches]
            combined = [content for batch in batch_results for content in batch]
            results.append(combined)
            idx += num_batches

        return results


class AnthropicGenerator:
    def __init__(self, concurrency: int):
        self._concurrency = concurrency
        self._client: AsyncAnthropic | None = None
        self._semaphore: asyncio.Semaphore | None = None

    async def __aenter__(self):
        self._client = AsyncAnthropic()
        self._semaphore = asyncio.Semaphore(self._concurrency)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._client:
            await self._client.close()

    async def _generate(self, **params: Any) -> str:
        if not self._client or not self._semaphore:
            raise RuntimeError("Generator not initialized")

        async with self._semaphore:
            completion = await self._client.messages.create(**params)
            return completion.content[-1].text

    async def generate_concurrently(
        self, prompts: list[list[dict[str, str]]], n: int = 1, **params: Any
    ) -> list[list[str]]:
        params = {k: v for k, v in params.items() if v is not None}
        if params.get("thinking_budget_tokens") is not None:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": params["thinking_budget_tokens"],
            }
            del params["thinking_budget_tokens"]

        expanded_prompts = [messages for messages in prompts for _ in range(n)]
        tasks = [
            self._generate(messages=messages, **params) for messages in expanded_prompts
        ]
        results = await tqdm_asyncio.gather(*tasks)

        return [results[i : i + n] for i in range(0, len(results), n)]
