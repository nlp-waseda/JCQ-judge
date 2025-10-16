import asyncio
import json
from argparse import ArgumentParser, Namespace
from itertools import chain
from pathlib import Path
from typing import Literal, Sequence

from utils import AnthropicGenerator, OpenAIGenerator, load_records


class Args(Namespace):
    provider: Literal["openai", "anthropic"]
    judge_model: str
    judge_model_id: str
    models: list[str]
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
        parser.add_argument("--judge-model", required=True)
        parser.add_argument("--judge-model-id", default="")
        parser.add_argument(
            "--models",
            nargs="+",
            required=True,
        )
        parser.add_argument("--max-completion-tokens", type=int)
        parser.add_argument("--max-tokens", default=4096, type=int)
        parser.add_argument("--concurrency", default=1, type=int)
        parser.add_argument("--data-dir", default="data", type=Path)
        return parser

    @classmethod
    def parse(cls, args: Sequence[str] | None = None) -> "Args":
        parser = cls._create_parser()
        parsed_args = parser.parse_args(args, namespace=cls())

        if not parsed_args.judge_model_id:
            parsed_args.judge_model_id = parsed_args.judge_model.strip("/").split("/")[
                -1
            ]
        return parsed_args

    @property
    def questions_path(self) -> Path:
        return self.data_dir / "test.jsonl"

    @property
    def judge_prompt_path(self) -> Path:
        return self.data_dir / "judge_prompt.txt"

    @property
    def answers_dir(self) -> Path:
        return self.data_dir / "answers"

    @property
    def judgements_dir(self) -> Path:
        return self.data_dir / "judgements" / self.judge_model_id


async def generate_and_save(args: Args) -> None:
    questions = load_records(args.questions_path)
    id2question = {record["id"]: record["question"] for record in questions}

    with open(args.judge_prompt_path, encoding="utf-8") as f:
        judge_prompt_template = f.read()

    for model in args.models:
        print(f"Judging {model} answers ...")
        answers_path = args.answers_dir / f"{model}.jsonl"
        answers = load_records(answers_path)

        judgements_path = args.judgements_dir / f"{model}.jsonl"
        if judgements_path.exists():
            raise FileExistsError(
                f"Judgements file '{judgements_path}' already exists."
            )

        prompts = []
        for record in answers:
            judge_prompt = judge_prompt_template.format(
                question=id2question[record["question_id"]], answer=record["answer"]
            )
            prompts.append([{"role": "user", "content": judge_prompt}])

        if args.provider == "openai":
            async with OpenAIGenerator(args.concurrency) as generator:
                outputs = await generator.generate_concurrently(
                    prompts=prompts,
                    model=args.judge_model,
                    max_completion_tokens=args.max_completion_tokens,
                    temperature=0,
                )
        else:
            async with AnthropicGenerator(args.concurrency) as generator:
                outputs = await generator.generate_concurrently(
                    prompts=prompts,
                    model=args.judge_model,
                    max_tokens=args.max_tokens,
                    temperature=0,
                )
        flat_outputs = list(chain.from_iterable(outputs))

        judgements_path.parent.mkdir(parents=True, exist_ok=True)
        with open(judgements_path, "w", encoding="utf-8") as f:
            for record, output in zip(answers, flat_outputs):
                result = {"answer_id": record["id"], "judgement": output}
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> None:
    args = Args.parse()

    asyncio.run(generate_and_save(args))


if __name__ == "__main__":
    main()
