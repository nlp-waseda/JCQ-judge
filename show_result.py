import argparse
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def save_radar_chart(df, file):
    fig = go.Figure()

    for index, row in df.iterrows():
        values = list(row) + [row[df.columns[0]]]
        categories = list(df.columns) + [df.columns[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                mode="lines+markers",
                name=index,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(size=14)),
            angularaxis=dict(
                tickfont=dict(size=14),
                rotation=90,
                direction="clockwise",
            ),
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=14),
            x=1.1,
            y=1.0,
        ),
        margin=dict(l=100, r=120, t=20, b=20),
        width=800,
        height=600,
    )

    os.makedirs(Path(file).parent, exist_ok=True)
    fig.write_image(file, scale=2)
    print(f"\nSaved radar chart at {file}")


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

    df = pd.merge(questions, judgements, left_on="id", right_on="question_id")
    df["mean"] = df[["fluency", "flexibility", "originality", "elaboration"]].mean(
        axis=1
    )
    df = df[
        [
            "task",
            "model",
            "fluency",
            "flexibility",
            "originality",
            "elaboration",
            "mean",
        ]
    ]

    radar_chart_directory = "data/model_judgement/radar_chart"

    print("\n########## Model and Criterion ##########")
    result_model_criterion = df.groupby("model")[
        ["fluency", "flexibility", "originality", "elaboration", "mean"]
    ].mean()
    result_model_criterion.index.name = None
    result_model_criterion.columns = result_model_criterion.columns.str.capitalize()
    print(result_model_criterion.round(2))

    save_radar_chart(
        result_model_criterion.drop(columns="Mean"),
        f"{radar_chart_directory}/{args.judge_model}_model_criterion.png",
    )

    print("\n########## Model and Task ##########")
    result_model_task = df.pivot_table(
        values="mean", index="model", columns="task", aggfunc="mean"
    )
    result_model_task = result_model_task[
        [
            "unusual uses",
            "consequences",
            "just suppose",
            "situation",
            "common problem",
            "improvement",
            "imaginative stories",
        ]
    ]
    result_model_task["all"] = result_model_criterion["Mean"]
    result_model_task.index.name = None
    result_model_task.columns.name = None
    result_model_task.columns = result_model_task.columns.str.capitalize()
    print(result_model_task.round(2))

    save_radar_chart(
        result_model_task.drop(columns="All"),
        f"{radar_chart_directory}/{args.judge_model}_model_task.png",
    )

    print("\n########## Task and Criterion ##########")
    result_task_criterion = df.groupby("task")[
        ["fluency", "flexibility", "originality", "elaboration", "mean"]
    ].mean()
    result_task_criterion = result_task_criterion.loc[
        [
            "unusual uses",
            "consequences",
            "just suppose",
            "situation",
            "common problem",
            "improvement",
            "imaginative stories",
        ]
    ]
    result_task_criterion.index.name = None
    result_task_criterion.index = result_task_criterion.index.str.capitalize()
    result_task_criterion.columns = result_task_criterion.columns.str.capitalize()
    print(result_task_criterion.round(2))

    save_radar_chart(
        result_task_criterion.drop(columns="Mean"),
        f"{radar_chart_directory}/{args.judge_model}_task_criterion.png",
    )
