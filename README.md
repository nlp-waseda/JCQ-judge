# JCQ-judge

Japanese Creativity Questions (JCQ)に対する LLM による回答と、LLM-as-a-judge による評価を行います。「流暢性」、「柔軟性」、「独創性」、「精緻性」の 4 指標を 1 ～ 5 のスケールで評価します。データセットや評価指標の詳細については[Huggingface のリポジトリ](https://huggingface.co/datasets/nlp-waseda/JCQ)や[論文](https://www.anlp.jp/proceedings/annual_meeting/2025/pdf_dir/C1-2.pdf)を参照してください。ローカルでモデルを動かす際は vLLM を用います。CUDA 12.9.1 で動作確認を行いました。それ以外のバージョンでは依存パッケージの調整が必要な場合があります。

## インストール

[uv](https://docs.astral.sh/uv/)を使用します。

```
git clone https://github.com/nlp-waseda/JCQ-judge.git
cd JCQ-judge
uv sync
```

## 回答と評価

### ステップ 1. 回答の生成

#### ローカルモデル

```
uv run src/gen_local_answers.py --model [MODEL]
```

引数

- `--model`：モデルの Hugging Face のリポジトリ ID またはローカルフォルダを指定してください。
- `--model-id`：モデルの識別名です。指定しなかった場合は`--model`で指定された名前（スラッシュ区切りの最後の部分）が使われます。
- `--num-choices`：1 つの質問に対する生成数です。デフォルト値は`1`。
- `--tensor-parallel-size`：デフォルト値は`1`。
- `--repetition-penalty`：デフォルト値は`1.0`。
- `--temperature`：デフォルト値は`1.0`。
- `--top-p`：デフォルト値は`1.0`。
- `--top-k`：デフォルト値は`0`（全トークン）。
- `--max-tokens`：デフォルト値は`4096`。
- `--max-model-len`：デフォルト値は`4096`。
- `--max-num-batched-tokens`：デフォルト値なし。
- `--max-num-seqs`：デフォルト値なし。
- `--data-dir`：データディレクトリです。デフォルト値は`data`。

`data/answers/[MODEL-ID].jsonl`にモデルの回答が保存されます。

例

```
uv run src/gen_local_answers.py --model llm-jp/llm-jp-3-3.7b-instruct3
```

#### API

```
export OPENAI_API_KEY=[OPENAI_API_KEY]
export ANTHROPIC_API_KEY=[ANTHROPIC_API_KEY]

uv run src/gen_api_answers.py \
    --provider [PROVIDER] \
    --model [MODEL]
```

必要に応じて環境変数`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`を設定してください。

引数

- `--provider`：使う API のプロバイダ （`openai`か`anthropic`）を指定してください。
- `--model`：モデルを指定してください。
- `--model-id`：モデルの識別名です。指定しなかった場合は`--model`で指定された名前が使われます。
- `--num-choices`：1 つの質問に対する生成数です。デフォルト値は`1`。
- `--temperature`：デフォルト値は`1.0`。
- `--reasoning-effort`：OpenAI API 専用の引数です。デフォルト値なし。
- `--thinking-budget-tokens`：Anthropic API 専用の引数です。デフォルト値なし。
- `--top-p`：デフォルト値なし。
- `--top-k`：Anthropic API 専用の引数です。デフォルト値なし。
- `--max-completion-tokens`：OpenAI API 専用の引数です。デフォルト値なし。
- `--max-tokens`：Anthropic API 専用の引数です。デフォルト値は`4096`。
- `--concurrency`：並行実行数を指定します。デフォルト値は`1`。※ API のレート制限に注意してください。
- `--data-dir`：データディレクトリです。デフォルト値は`data`。

`data/answers/[MODEL-ID].jsonl`にモデルの回答が保存されます。

例

```
export OPENAI_API_KEY=XXXXXX

uv run src/gen_api_answers.py \
    --provider openai \
    --model gpt-4o-mini \
    --concurrency 10
```

### ステップ 2. 評価

```
export OPENAI_API_KEY=[OPENAI_API_KEY]
export ANTHROPIC_API_KEY=[ANTHROPIC_API_KEY]

uv run src/gen_api_judgements.py \
    --provider [PROVIDER] \
    --judge-model [JUDGE-MODEL] \
    --models [MODELS]
```

必要に応じて環境変数`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`を設定してください。

引数

- `--provider`：使う API のプロバイダ （`openai`か`anthropic`）を指定してください。
- `--judge-model`：評価モデルを指定してください。
- `--judge-model-id`：評価モデルの識別名です。指定しなかった場合は`--judge-model`で指定された名前が使われます。
- `--models`：評価対象のモデルの`[MODEL-ID]`を空白区切りで指定してください。
- `--max-completion-tokens`：OpenAI API 専用の引数です。デフォルト値なし。
- `--max-tokens`：Anthropic API 専用の引数です。デフォルト値は`4096`。
- `--concurrency`：並行実行数を指定します。デフォルト値は`1`。※ API のレート制限に注意してください。
- `--data-dir`：データディレクトリです。デフォルト値は`data`。

`[MODELS]`で指定されたモデルの回答に対する評価が`data/judgements/[JUDGE-MODEL-ID]/[MODEL-ID].jsonl`に保存されます。

例

```
export OPENAI_API_KEY=XXXXXX

uv run src/gen_api_judgements.py \
    --provider openai \
    --judge-model gpt-4o-mini \
    --models gpt-4o-mini llm-jp-3-3.7b-instruct3 \
    --concurrency 5
```

### ステップ 3. 結果の表示

```
uv run src/show_results.py \
    --judge-model [JUDGE-MODEL] \
    --models [MODELS]
```

引数

- `--judge-model`：評価モデルの`[JUDGE-MODEL-ID]`を指定してください。
- `--models`：評価対象のモデルの`[MODEL-ID]`を空白区切りで指定してください。
- `--compare`：指定すると 2 つのモデルのタスク・指標ごとの比較結果が表示されます。`[MODELS]`が 2 つの場合のみ指定できます。
- `--data-dir`：データディレクトリです。デフォルト値は`data`。

`[JUDGE-MODEL]`で指定された評価モデルによる`[MODELS]`で指定されたモデルの回答に対する評価結果が表示されます。

例

```
uv run src/show_results.py \
    --judge-model gpt-4o-mini \
    --models gpt-4o-mini llm-jp-3-3.7b-instruct3
```

## 評価結果

いくつかのモデルの評価結果を記載します。生成時の temperature は 1.0、評価モデルは gpt-4o-2024-08-06 です。

### モデルと指標

|                                                  | 流暢性   | 柔軟性   | 独創性   | 精緻性   | 平均     |
| ------------------------------------------------ | -------- | -------- | -------- | -------- | -------- |
| gpt-4o-2024-08-06                                | 4.10     | **4.28** | 2.73     | 3.47     | 3.64     |
| claude-3-5-sonnet-20241022                       | **4.29** | 4.04     | 2.73     | 2.87     | 3.48     |
| cyberagent/calm3-22b-chat                        | 4.16     | 4.18     | **2.87** | **3.86** | **3.76** |
| llm-jp/llm-jp-3-13b-instruct                     | 3.74     | 3.79     | 2.65     | 3.45     | 3.41     |
| tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1 | 3.91     | 3.45     | 2.34     | 2.79     | 3.12     |

### モデルとタスク

|                                                  | 非通常使用 | 結果     | 仮定     | 状況     | 一般的問題 | 改善     | 想像的物語 |
| ------------------------------------------------ | ---------- | -------- | -------- | -------- | ---------- | -------- | ---------- |
| gpt-4o-2024-08-06                                | **3.97**   | 3.69     | 3.83     | 3.28     | 3.48       | **4.01** | 3.25       |
| claude-3-5-sonnet-20241022                       | 3.73       | 3.42     | 3.80     | 3.08     | **3.61**   | 3.80     | 2.93       |
| cyberagent/calm3-22b-chat                        | 3.84       | 3.92     | **3.91** | **3.73** | 3.45       | 4.00     | **3.50**   |
| llm-jp/llm-jp-3-13b-instruct                     | 3.08       | **3.92** | 3.52     | 3.69     | 3.00       | 3.64     | 3.01       |
| tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.1 | 3.28       | 3.33     | 3.39     | 2.80     | 3.08       | 3.45     | 2.54       |

### タスクと指標

|            | 流暢性 | 柔軟性 | 独創性 | 精緻性 | 平均 |
| ---------- | ------ | ------ | ------ | ------ | ---- |
| 非通常使用 | 4.50   | 4.13   | 2.92   | 2.78   | 3.58 |
| 結果       | 4.00   | 4.31   | 2.67   | 3.64   | 3.65 |
| 仮定       | 4.58   | 4.43   | 2.64   | 3.11   | 3.69 |
| 状況       | 3.30   | 4.03   | 2.57   | 3.38   | 3.32 |
| 一般的問題 | 3.98   | 3.85   | 2.01   | 3.46   | 3.32 |
| 改善       | 4.71   | 4.51   | 2.72   | 3.17   | 3.78 |
| 想像的物語 | 3.22   | 2.36   | 3.12   | 3.49   | 3.05 |
