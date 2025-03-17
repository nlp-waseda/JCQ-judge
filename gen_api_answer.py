import argparse
import concurrent.futures
import json
from pathlib import Path
from threading import Lock

import pandas as pd
from tqdm import tqdm

from common import load_data, chat_completion_anthropic, chat_completion_openai

file_lock = Lock()


def get_answer(
    record,
    api,
    model,
    max_tokens,
    temperature,
    top_p,
    top_k,
    answer_file,
    system_prompt,
):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": record["question"]})

    if api == "anthropic":
        output = chat_completion_anthropic(
            model, messages, max_tokens, temperature, top_p, top_k
        )
    else:
        output = chat_completion_openai(model, messages, max_tokens, temperature, top_p)

    answer = {"question_id": record["id"], "answer": output}

    Path(answer_file).parent.mkdir(parents=True, exist_ok=True)
    with file_lock:
        with open(answer_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(answer, ensure_ascii=False) + "\n")


def reorg_answer_file(answer_file):
    answers = pd.read_json(answer_file, lines=True)
    answers = answers.sort_values("question_id")
    answers.to_json(answer_file, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True, choices=["openai", "anthropic"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--max_tokens", default=4096, type=int)
    parser.add_argument("--parallel", default=1, type=int)
    parser.add_argument("--system_prompt")
    args = parser.parse_args()

    data = load_data("data/test.jsonl")

    answer_file = f"data/model_answer/{args.model}.jsonl"

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for record in data:
            future = executor.submit(
                get_answer,
                record,
                args.api,
                args.model,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.top_k,
                answer_file,
                args.system_prompt,
            )
            futures.append(future)

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)
