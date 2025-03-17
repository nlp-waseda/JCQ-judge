import argparse
import json
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams

from common import load_data


def get_model_answers(
    model_path,
    data,
    answer_file,
    batch_size,
    temperature,
    top_p,
    top_k,
    max_tokens,
    tensor_parallel_size,
    system_prompt,
):
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens
    )

    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i : i + batch_size]

        prompts = []
        for record in batch:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": record["question"]})

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        batch_answers = [
            {
                "question_id": record["id"],
                "answer": output.outputs[0].text,
            }
            for record, output in zip(batch, outputs)
        ]

        Path(answer_file).parent.mkdir(parents=True, exist_ok=True)
        with open(answer_file, "a", encoding="utf-8") as f:
            f.writelines(
                [
                    json.dumps(answer, ensure_ascii=False) + "\n"
                    for answer in batch_answers
                ]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--max_tokens", default=4096, type=int)
    parser.add_argument("--tensor_parallel_size", default=1, type=int)
    parser.add_argument("--system_prompt")
    args = parser.parse_args()

    data = load_data("data/test.jsonl")

    answer_file = f"data/model_answer/{args.model_id}.jsonl"

    get_model_answers(
        model_path=args.model_path,
        data=data,
        answer_file=answer_file,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        system_prompt=args.system_prompt,
    )
