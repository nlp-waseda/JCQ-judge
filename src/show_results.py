import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Sequence, cast

import pandas as pd

from utils import load_records

pd.set_option("display.width", None)

CRITERIA = ["fluency", "flexibility", "originality", "elaboration"]
TASKS = [
    "unusual uses",
    "consequences",
    "just suppose",
    "situation",
    "common problem",
    "improvement",
    "imaginative stories",
]
PATTERN = re.compile(
    r"流暢性:\s*\[?([1-5])\]?\s*柔軟性:\s*\[?([1-5])\]?\s*独創性:\s*\[?([1-5])\]?\s*精緻性:\s*\[?([1-5])\]?"
)


class Args(Namespace):
    judge_model: str
    models: list[str]
    compare: bool
    data_dir: Path

    @classmethod
    def _create_parser(cls) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument("--judge-model", required=True)
        parser.add_argument(
            "--models",
            nargs="+",
            required=True,
        )
        parser.add_argument("--compare", action="store_true")
        parser.add_argument("--data-dir", default="data", type=Path)
        return parser

    @classmethod
    def parse(cls, args: Sequence[str] | None = None) -> "Args":
        parser = cls._create_parser()
        parsed_args = parser.parse_args(args, namespace=cls())

        if parsed_args.compare and len(parsed_args.models) != 2:
            raise ValueError(
                "--compare mode requires exactly 2 models to be specified with --models."
            )
        return parsed_args

    @property
    def questions_path(self) -> Path:
        return self.data_dir / "test.jsonl"

    @property
    def answers_dir(self) -> Path:
        return self.data_dir / "answers"

    @property
    def judgements_dir(self) -> Path:
        return self.data_dir / "judgements" / self.judge_model


def _extract_scores(text: str) -> list[int] | list[None]:
    match = PATTERN.search(text)
    if match:
        scores = list(map(int, match.groups()))
    else:
        scores = [None] * len(CRITERIA)
    return scores


def _build_dataframe(args: Args) -> pd.DataFrame:
    questions = load_records(args.questions_path)
    question_id2task: dict[int, str] = {
        record["id"]: record["task"] for record in questions
    }
    data: list[dict[str, int | str | None]] = []
    for model in args.models:
        answers = load_records(args.answers_dir / f"{model}.jsonl")
        answer_id2question_id: dict[int, int] = {
            record["id"]: record["question_id"] for record in answers
        }
        judgements = load_records(args.judgements_dir / f"{model}.jsonl")
        for record in judgements:
            question_id = answer_id2question_id[record["answer_id"]]
            task = question_id2task[question_id]
            scores = _extract_scores(record["judgement"])
            if scores[0] is None:
                print(
                    f"Ignoring invalid judgement: question_id={question_id}, answer_id={record['answer_id']}, judgement={record['judgement']}"
                )
                continue

            data.append(
                {
                    "question_id": question_id,
                    "task": task,
                    "model": model,
                    **dict(zip(CRITERIA, scores)),
                }
            )

    df = pd.DataFrame(data)
    df["count"] = df.groupby(["question_id", "model"]).cumcount() + 1

    return df


def show_summary(args: Args) -> None:
    df = _build_dataframe(args)
    df["mean"] = df[CRITERIA].mean(axis=1)

    def format_mean_std(series: pd.Series) -> str:
        mean = series.mean()
        std = series.std()
        if pd.isna(std):
            return f"{mean:.2f}"
        return f"{mean:.2f} (±{std:.2f})"

    print("\n########## Model and Criterion ##########")
    df_model_criterion = (
        df.groupby(["model", "count"])[CRITERIA + ["mean"]]
        .mean()
        .groupby("model")
        .agg(format_mean_std)
        .reindex(args.models)
    )
    df_model_criterion.index.name = None
    df_model_criterion.columns = df_model_criterion.columns.str.capitalize()
    print(df_model_criterion)

    print("\n########## Model and Task ##########")
    df_model_task = (
        df.groupby(["model", "task", "count"])["mean"]
        .mean()
        .groupby(["model", "task"])
        .agg(format_mean_std)
        .unstack()
        .reindex(index=args.models, columns=TASKS)
    )
    df_model_task["all"] = df_model_criterion["Mean"]
    df_model_task.index.name = None
    df_model_task.columns = df_model_task.columns.str.capitalize()
    print(df_model_task)

    print("\n########## Task and Criterion ##########")
    df_task_criterion = (
        df.groupby(["task", "count"])[CRITERIA + ["mean"]]
        .mean()
        .groupby("task")
        .agg(format_mean_std)
        .reindex(TASKS)
    )
    df_task_criterion.index.name = None
    df_task_criterion.columns = df_task_criterion.columns.str.capitalize()
    print(df_task_criterion)


def show_comparison(args: Args) -> None:
    df = _build_dataframe(args)

    agg_df: pd.DataFrame = (
        df.groupby(["task", "model", "count"])[CRITERIA]
        .mean()
        .groupby(["task", "model"])
        .agg(["mean", "std"])
    )

    results = []
    for task in TASKS:
        for criterion in CRITERIA:
            row = {}
            for model in args.models:
                mean_val = agg_df.loc[(task, model), (criterion, "mean")]
                std_val = agg_df.loc[(task, model), (criterion, "std")]
                if pd.isna(std_val):
                    row[model] = f"{mean_val:.2f}"
                else:
                    row[model] = f"{mean_val:.2f} (±{std_val:.2f})"

            mean1 = cast(float, agg_df.loc[(task, args.models[0]), (criterion, "mean")])
            mean2 = cast(float, agg_df.loc[(task, args.models[1]), (criterion, "mean")])
            diff = mean2 - mean1
            row["diff"] = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"

            results.append(row)

    df_compare = pd.DataFrame(
        results,
        index=[
            f"{task.capitalize()}, {criterion.capitalize()}"
            for task in TASKS
            for criterion in CRITERIA
        ],
    )

    print(df_compare)


def main() -> None:
    args = Args.parse()

    if not args.compare:
        show_summary(args)
    else:
        show_comparison(args)


if __name__ == "__main__":
    main()
