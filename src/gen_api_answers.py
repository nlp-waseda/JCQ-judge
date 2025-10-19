import asyncio
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal, Sequence

from utils import AnthropicGenerator, OpenAIGenerator, load_records, save_answers


class Args(Namespace):
    provider: Literal["openai", "anthropic"]
    model: str
    model_id: str | None
    num_choices: int
    temperature: float
    reasoning_effort: str | None
    thinking_budget_tokens: int | None
    top_p: float | None
    top_k: int | None
    max_completion_tokens: int | None
    max_tokens: int
    concurrency: int
    data_dir: Path

    @classmethod
    def _create_parser(cls) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument(
            "--provider", required=True, choices=["openai", "anthropic"]
        )
        parser.add_argument("--model", required=True)
        parser.add_argument("--model-id")
        parser.add_argument("--num-choices", default=1, type=int)
        parser.add_argument("--temperature", default=1.0, type=float)
        parser.add_argument("--reasoning-effort")
        parser.add_argument("--thinking-budget-tokens", type=int)
        parser.add_argument("--top-p", type=float)
        parser.add_argument("--top-k", type=int)
        parser.add_argument("--max-completion-tokens", type=int)
        parser.add_argument("--max-tokens", default=4096, type=int)
        parser.add_argument("--concurrency", default=1, type=int)
        parser.add_argument("--data-dir", default="data", type=Path)
        return parser

    @classmethod
    def parse(cls, args: Sequence[str] | None = None) -> "Args":
        parser = cls._create_parser()
        parsed_args = parser.parse_args(args, namespace=cls())

        if parsed_args.model_id is None:
            parsed_args.model_id = parsed_args.model.strip("/").split("/")[-1]
        return parsed_args

    @property
    def questions_path(self) -> Path:
        return self.data_dir / "test.jsonl"

    @property
    def answers_path(self) -> Path:
        return self.data_dir / "answers" / f"{self.model_id}.jsonl"


async def generate_and_save(args: Args) -> None:
    questions = load_records(args.questions_path)
    prompts = [
        [{"role": "user", "content": record["question"]}] for record in questions
    ]

    if args.answers_path.exists():
        raise FileExistsError(f"Answer file '{args.answers_path}' already exists")

    if args.provider == "openai":
        async with OpenAIGenerator(args.concurrency) as generator:
            answers = await generator.generate_concurrently(
                prompts=prompts,
                model=args.model,
                max_completion_tokens=args.max_completion_tokens,
                n=args.num_choices,
                reasoning_effort=args.reasoning_effort,
                temperature=args.temperature,
                top_p=args.top_p,
            )
    else:
        async with AnthropicGenerator(args.concurrency) as generator:
            answers = await generator.generate_concurrently(
                prompts=prompts,
                n=args.num_choices,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                thinking_budget_tokens=args.thinking_budget_tokens,
                top_k=args.top_k,
                top_p=args.top_p,
            )

    save_answers(args.answers_path, questions, answers)


def main() -> None:
    args = Args.parse()

    asyncio.run(generate_and_save(args))


if __name__ == "__main__":
    main()
