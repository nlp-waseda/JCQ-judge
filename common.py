import json

import anthropic
import openai


def load_data(file):
    with open(file, encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def chat_completion_openai(model, messages, max_completion_tokens, temperature, top_p):
    completion = openai.OpenAI().chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    return completion.choices[0].message.content


def chat_completion_anthropic(model, messages, max_tokens, temperature, top_p, top_k):
    completion = anthropic.Anthropic().messages.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    return completion.content[0].text
