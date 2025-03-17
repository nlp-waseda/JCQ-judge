import argparse
import concurrent.futures
import json
import os
import re
from pathlib import Path
from threading import Lock

import pandas as pd
from tqdm import tqdm

from common import chat_completion_anthropic, chat_completion_openai

file_lock = Lock()


def get_judgement(
    record,
    judge_prompt,
    api,
    judge_model,
    judge_file,
    system_prompt,
):
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": judge_prompt.format(
                question=record["question"],
                answer=record["answer"],
            ),
        }
    )

    if api == "anthropic":
        output = chat_completion_anthropic(judge_model, messages, 64, 0, 1.0, -1)
    else:
        output = chat_completion_openai(judge_model, messages, 64, 0, 1.0)

    pattern = r"流暢性:\s*\[?([1-5])\]?\s*柔軟性:\s*\[?([1-5])\]?\s*独創性:\s*\[?([1-5])\]?\s*精緻性:\s*\[?([1-5])\]?"
    match = re.search(pattern, output)

    if match:
        scores = list(map(int, match.groups()))
    else:
        print(
            f"Invalid judgement. model: {record['model']}, question_id: {record['question_id']}, {output}"
        )
        scores = [None] * 4

    judgement = {
        "model": record["model"],
        "question_id": record["question_id"],
        "fluency": scores[0],
        "flexibility": scores[1],
        "originality": scores[2],
        "elaboration": scores[3],
    }

    os.makedirs(Path(judge_file).parent, exist_ok=True)
    with file_lock:
        with open(judge_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(judgement, ensure_ascii=False) + "\n")


def reorg_judge_file(judge_file):
    judgements = pd.read_json(judge_file, lines=True)
    judgements = judgements.sort_values(["model", "question_id"])
    judgements.to_json(judge_file, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True, choices=["openai", "anthropic"])
    parser.add_argument("--judge_model", required=True)
    parser.add_argument("--model_list", required=True, nargs="+")
    parser.add_argument("--parallel", default=1, type=int)
    parser.add_argument("--system_prompt")
    args = parser.parse_args()

    questions = pd.read_json("data/test.jsonl", lines=True)

    answers = []
    for model in args.model_list:
        model_answers = pd.read_json(f"data/model_answer/{model}.jsonl", lines=True)
        model_answers["model"] = model
        answers.append(model_answers)
    answers = pd.concat(answers)

    qa = pd.merge(questions, answers, left_on="id", right_on="question_id")
    qa = qa.drop(columns="id")

    with open("data/judge_prompt.txt", encoding="utf-8") as f:
        judge_prompt = f.read()

    judge_file = f"data/model_judgement/{args.judge_model}.jsonl"

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for record in qa.to_dict(orient="records"):
            future = executor.submit(
                get_judgement,
                record,
                judge_prompt,
                args.api,
                args.judge_model,
                judge_file,
                args.system_prompt,
            )
            futures.append(future)

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_judge_file(judge_file)
