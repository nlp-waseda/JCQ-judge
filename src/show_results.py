import re
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Sequence, cast

import pandas as pd

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
    r"流暢性:\s*\[?([1-5])\]?\s*"
    r"柔軟性:\s*\[?([1-5])\]?\s*"
    r"独創性:\s*\[?([1-5])\]?\s*"
    r"精緻性:\s*\[?([1-5])\]?"
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
                "--compare mode requires exactly 2 models to be specified with --models"
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


def _build_dataframe(args: Args) -> pd.DataFrame:
    df_questions = pd.read_json(args.data_dir / "test.jsonl", lines=True)
    df_questions = df_questions.rename(columns={"id": "question_id"})
    dfs = []
    for model in args.models:
        df_answers = pd.read_json(args.answers_dir / f"{model}.jsonl", lines=True)
        df_answers = df_answers.rename(columns={"id": "answer_id"})
        df_judgements = pd.read_json(args.judgements_dir / f"{model}.jsonl", lines=True)
        df_merged = df_questions.merge(df_answers).merge(df_judgements)

        df_extracted = df_merged["judgement"].str.extract(PATTERN)
        invalid_mask = df_extracted.isna().any(axis=1)
        for idx in df_extracted[invalid_mask].index:
            row = df_merged.loc[idx]
            print(
                f"Ignoring invalid judgement: question_id={row['question_id']}, "
                f"answer_id={row['answer_id']}, judgement={row['judgement']}"
            )
        df_merged[CRITERIA] = df_extracted.apply(pd.to_numeric, errors="coerce")
        df_merged = df_merged[~invalid_mask]
        df_merged["model"] = model
        df_merged = df_merged[["question_id", "task", "model", *CRITERIA]]
        dfs.append(df_merged)
    df = pd.concat(dfs)
    df["count"] = df.groupby(["question_id", "model"]).cumcount() + 1

    return df


def main(args: Args) -> None:
    pd.set_option("display.width", None)

    df = _build_dataframe(args)

    if not args.compare:
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

    else:
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

                mean1 = cast(
                    float, agg_df.loc[(task, args.models[0]), (criterion, "mean")]
                )
                mean2 = cast(
                    float, agg_df.loc[(task, args.models[1]), (criterion, "mean")]
                )
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


if __name__ == "__main__":
    args = Args.parse()

    main(args)
