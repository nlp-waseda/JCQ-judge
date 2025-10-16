import asyncio
import json
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


def load_records(file: str | Path) -> list[dict[str, Any]]:
    with open(file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return records


def save_answers(
    path: Path, records: list[dict[str, Any]], outputs: list[list[str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        answer_id = 1
        for record, output_list in zip(records, outputs):
            for answer in output_list:
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
        tasks = [self._generate(messages=messages, **params) for messages in prompts]
        return await tqdm_asyncio.gather(*tasks)


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
