from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Sequence

from utils import load_records, save_answers
from vllm import LLM, SamplingParams
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


class Args(Namespace):
    model: str
    model_id: str | None
    num_choices: int
    tensor_parallel_size: int
    repetition_penalty: float
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    max_model_len: int
    max_num_batched_tokens: int | None
    max_num_seqs: int | None
    data_dir: Path

    @classmethod
    def _create_parser(cls) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument("--model", required=True)
        parser.add_argument("--model-id")
        parser.add_argument("--num-choices", default=1, type=int)
        parser.add_argument("--tensor-parallel-size", default=1, type=int)
        parser.add_argument("--repetition-penalty", default=1.0, type=float)
        parser.add_argument("--temperature", default=1.0, type=float)
        parser.add_argument("--top-p", default=1.0, type=float)
        parser.add_argument("--top-k", default=0, type=int)
        parser.add_argument("--max-tokens", default=4096, type=int)
        parser.add_argument("--max-model-len", default=4096, type=int)
        parser.add_argument("--max-num-batched-tokens", type=int)
        parser.add_argument("--max-num-seqs", type=int)
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


def generate_and_save(args: Args) -> None:
    questions = load_records(args.questions_path)
    prompts: list[list[ChatCompletionMessageParam]] = [
        [{"role": "user", "content": record["question"]}] for record in questions
    ]

    if args.answers_path.exists():
        raise FileExistsError(f"Answer file '{args.answers_path}' already exists")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
    )
    sampling_params = SamplingParams(
        n=args.num_choices,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    outputs = llm.chat(prompts, sampling_params)

    answers = [[choice.text for choice in output.outputs] for output in outputs]

    save_answers(args.answers_path, questions, answers)


def main() -> None:
    args = Args.parse()

    generate_and_save(args)


if __name__ == "__main__":
    main()
