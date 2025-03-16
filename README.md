# JCQ-judge

Japanese Creativity Questions (JCQ)に対する LLM による回答と、LLM-as-a-judge による評価を行います。「流暢性」、「柔軟性」、「独創性」、「精緻性」の 4 指標を 1 ～ 5 のスケールで評価します。データセットや評価指標の詳細については[Huggingface のリポジトリ](https://huggingface.co/datasets/nlp-waseda/JCQ)や[論文](https://www.anlp.jp/proceedings/annual_meeting/2025/pdf_dir/C1-2.pdf)を参照してください。ローカルでモデルを動かす際は vLLM を用います。Python 3.9.18, CUDA 11.8.0 で動作確認を行いました。それ以外のバージョンでは依存パッケージの調整が必要な場合があります。

## インストール

```
git clone https://github.com/nlp-waseda/JCQ-judge.git
pip install -r requirements.txt
```

## 回答と評価

### ステップ 1. 回答の生成

#### ローカルモデル

```
python gen_model_answer.py \
    --model_path [MODEL_PATH] \
    --model_id [MODEL_ID]
```

- `[MODEL_PATH]`はモデルのパスです。ローカルフォルダまたは Hugging Face のリポジトリ ID を指定してください。
- `[MODEL_ID]`はモデルの識別名です。一意の名前を付けてください。
- その他以下の項目を指定することができます。
  - `--batch_size` (デフォルト値：64)
  - `--temperature` (デフォルト値：1.0)
  - `--top_p` (デフォルト値：1.0)
  - `--top_k` (デフォルト値：-1)
  - `--max_tokens` (デフォルト値：4096)
  - `--tensor_parallel_size` (デフォルト値：1)
  - `--system_prompt`: モデルに対してシステムメッセージを指定できます（デフォルト値なし、例："あなたは親切な AI アシスタントです。"）

例

```
python gen_model_answer.py \
    --model_path llm-jp/llm-jp-3-1.8b-instruct3 \
    --model_id llm-jp-3-1.8b-instruct3
```

#### API

```
export OPENAI_API_KEY=[OPENAI_API_KEY]
export ANTHROPIC_API_KEY=[ANTHROPIC_API_KEY]

python gen_api_answer.py \
    --api [API] \
    --model [MODEL]
```

- `[OPENAI_API_KEY]`は OpenAI の API キーです。使う場合は環境変数`OPENAI_API_KEY`にセットしてください。
- `[ANTHROPIC_API_KEY]`は Anthropic の API キーです。使う場合は環境変数`ANTHROPIC_API_KEY`にセットしてください。
- `[API]`で使う API を指定してください。`openai`か`anthropic`です。
- `[MODEL]`でモデルを指定してください。
- その他以下の項目を指定することができます。
  - `--temperature` (デフォルト値：1.0)
  - `--top_p` (デフォルト値：1.0)
  - `--top_k` (デフォルト値：-1, OpenAI API では指定できない)
  - `--max_tokens` (デフォルト値：4096)
  - `--parallel`: 並列実行数を指定します（デフォルト値：1）※ API のレート制限に注意してください。
  - `--system_prompt`: モデルに対してシステムメッセージを指定できます（デフォルト値なし、例："あなたは親切な AI アシスタントです。"）

例

```
export OPENAI_API_KEY=XXXXXX

python gen_api_answer.py \
    --api openai \
    --model gpt-4o-mini \
    --parallel 10
```

### ステップ 2. 評価

```
export OPENAI_API_KEY=[OPENAI_API_KEY]
export ANTHROPIC_API_KEY=[ANTHROPIC_API_KEY]

python gen_api_judgement.py \
    --api [API] \
    --judge_model [JUDGE_MODEL] \
    --model_list [MODEL_LIST]
```

- `[OPENAI_API_KEY]`は OpenAI の API キーです。使う場合は環境変数`OPENAI_API_KEY`にセットしてください。
- `[ANTHROPIC_API_KEY]`は Anthropic の API キーです。使う場合は環境変数`ANTHROPIC_API_KEY`にセットしてください。
- `[API]`で使う API を指定してください。`openai`か`anthropic`です。
- `[JUDGE_MODEL]`で評価モデルを指定してください。
- `[MODEL_LIST]`で評価対象のモデルを空白区切りで指定してください。
- その他以下の項目を指定することができます。
  - `--parallel`: 並列実行数を指定します（デフォルト値：1）※ API のレート制限に注意してください。
  - `--system_prompt`: モデルに対してシステムメッセージを指定できます（デフォルト値なし、例："あなたは親切な AI アシスタントです。"）

例

```
python gen_api_judgement.py \
    --api openai \
    --judge_model gpt-4o-mini \
    --model_list llm-jp-3-1.8b-instruct3 gpt-4o-mini \
    --parallel 5
```

### ステップ 3. 結果の表示

```
python show_result.py --judge_model [JUDGE_MODEL]
```

- `[JUDGE_MODEL]`で評価モデルを指定してください。
- その他以下の項目を指定することができます。
  - `--model_list` (デフォルト値なし)

結果が表示されます。また、レーダーチャートが`llm_judge/data/model_judgement/radar_chart/`ディレクトリに PNG 形式で保存されます。

例

```
python show_result.py --judge_model gpt-4o-mini
```

## 評価結果

いくつかのモデルの評価結果を記載します。評価モデルは gpt-4o-2024-08-06 です。

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
