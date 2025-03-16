import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_radar_chart_model_criterion(df, file):
    models = df.index.to_list()
    criteria = ["fluency", "flexibility", "originality", "elaboration"]

    angles = [n / len(criteria) * 2 * np.pi for n in range(len(criteria))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in models:
        values = df.loc[model].to_list()
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=model)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria)

    ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0))

    os.makedirs(Path(file).parent, exist_ok=True)
    plt.savefig(file, dpi=300, bbox_inches="tight")
    plt.close()


def save_radar_chart_model_task(df, file):
    models = df.index.to_list()
    tasks = [
        "unusual uses",
        "consequences",
        "just suppose",
        "situation",
        "common problem",
        "improvement",
        "imaginative stories",
    ]

    angles = [n / len(tasks) * 2 * np.pi for n in range(len(tasks))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in models:
        values = df.loc[model].to_list()
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=model)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tasks)

    ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0))

    os.makedirs(Path(file).parent, exist_ok=True)
    plt.savefig(file, dpi=300, bbox_inches="tight")
    plt.close()


def save_radar_chart_task_criterion(df, file):
    tasks = df.index.to_list()
    criteria = ["fluency", "flexibility", "originality", "elaboration"]

    angles = [n / len(criteria) * 2 * np.pi for n in range(len(criteria))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for task in tasks:
        values = df.loc[task].to_list()
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=task)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria)

    ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0))

    os.makedirs(Path(file).parent, exist_ok=True)
    plt.savefig(file, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model", required=True)
    parser.add_argument("--model_list", nargs="*")
    args = parser.parse_args()

    questions = pd.read_json("data/test.jsonl", lines=True)
    questions = questions[["id", "task"]]

    judgements = pd.read_json(
        f"data/model_judgement/{args.judge_model}.jsonl", lines=True
    )
    judgements = judgements.dropna()

    if args.model_list is not None:
        judgements = judgements[judgements["model"].isin(args.model_list)]

    qj = pd.merge(questions, judgements, left_on="id", right_on="question_id")
    qj.drop(columns="question_id")

    qj["mean"] = qj[["fluency", "flexibility", "originality", "elaboration"]].mean(
        axis=1
    )

    radar_chart_directory = "data/model_judgement/radar_chart"

    print("\n########## Model and Criterion ##########")
    result_model_criterion = qj.groupby("model")[
        ["fluency", "flexibility", "originality", "elaboration", "mean"]
    ].mean()
    result_model_criterion.index.name = None
    print(result_model_criterion.round(2))

    radar_chart_file = f"{radar_chart_directory}/{args.judge_model}_model_criterion.png"
    save_radar_chart_model_criterion(
        result_model_criterion.drop(columns="mean"), radar_chart_file
    )
    print(f"\nSaved radar chart at {radar_chart_file}")

    print("\n########## Model and Task ##########")
    result_model_task = qj.pivot_table(
        values="mean", index="model", columns="task", aggfunc="mean"
    )
    result_model_task["all"] = result_model_criterion["mean"]
    result_model_task.index.name = None
    result_model_task.columns.name = None
    print(result_model_task.round(2))

    radar_chart_file = f"{radar_chart_directory}/{args.judge_model}_model_task.png"
    save_radar_chart_model_task(result_model_task.drop(columns="all"), radar_chart_file)
    print(f"\nSaved radar chart at {radar_chart_file}")

    print("\n########## Task and Criterion ##########")
    result_task_criterion = qj.groupby("task")[
        ["fluency", "flexibility", "originality", "elaboration", "mean"]
    ].mean()
    result_task_criterion.index.name = None
    print(result_task_criterion.round(2))

    radar_chart_file = f"{radar_chart_directory}/{args.judge_model}_task_criterion.png"
    save_radar_chart_task_criterion(
        result_task_criterion.drop(columns="mean"), radar_chart_file
    )
    print(f"\nSaved radar chart at {radar_chart_file}")
