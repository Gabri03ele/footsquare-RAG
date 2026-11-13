# Footsquare RAG

A Retrieval-Augmented Generation (RAG) pipeline that answers multiple-choice questions about the rules
of **Footsquare**, an invented team sport. Because the sport does not exist, a language model cannot
answer from prior knowledge: every answer has to be grounded in the rulebook retrieved at query time.

**Result: 0.760 accuracy (19/25) on the Codabench evaluation.**

Built for the Natural Language Processing course of the MSc in Data Science at the University of Padova.

---

## The task

Footsquare is played on a circular pitch with raised baskets instead of goals. Its rulebook covers the
pitch (central circle, control ring, quadrant lines), scoring (baskets, a 5 m scoring arc worth 1 or 2
points), keeper zones, players and substitutions, match duration, restarts, fouls, cards and video review.

- **Knowledge base**: 49 short rules (`rules.json`), one or two sentences each.
- **Questions**: 25 multiple-choice questions (`questions_mcq_student.json`).
- **Evaluation**: predictions are submitted to Codabench as a CSV (`id`, `prediction`) and scored by accuracy.
  No labels are available locally.

## Pipeline

For each question the pipeline runs five steps:

| Step        | What happens                                                                                                                                     | Module            |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- |
| 1. Index    | The rules are split into chunks and embedded once, before any question is asked.                                                                 | `src/index.py`    |
| 2. Retrieve | The question is enriched with numbers and keywords from its options, embedded, and compared with every chunk; the 7 most similar rules are kept. | `src/retrieve.py` |
| 3. Generate | The 7 rules, the question and its options are given to the language model, which replies with the letter of the correct option.                  | `src/generate.py` |
| 4. Judge    | The same model checks whether the answer is supported by the rules and whether the rules were relevant to the question.                          | `src/judge.py`    |
| 5. Export   | The letters are saved in a CSV for evaluation.                                                                                                   | `src/export.py`   |

Below each step in detail.

### 1. Indexing — `src/index.py`

Each rule is split on sentence boundaries into chunks of at most 200 words, with a 40-word overlap
between consecutive chunks, and each chunk is embedded with
[`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
Embeddings are L2-normalized, so cosine similarity reduces to a dot product.

With the current rulebook every rule fits in a single chunk (49 rules → 49 chunks); the chunking keeps the
pipeline usable with a longer knowledge base.

### 2. Retrieval — `src/retrieve.py`

The question is augmented with the numbers and a fixed list of keywords found in the question and its
options, so that rules containing the same values (for example "5 m" or "2 points") rank higher.
The augmented query is embedded with the same model and the 7 most similar chunks are retrieved.
Duplicated terms are removed while preserving their order, so the query is identical across runs.

### 3. Generation — `src/generate.py`

The retrieved rules are numbered and inserted into a prompt together with the question and its options.
[`unsloth/gemma-3-1B-it`](https://huggingface.co/unsloth/gemma-3-1B-it) answers with greedy decoding
(at most 50 new tokens), so the output is deterministic. The chosen letter is extracted with three
fallbacks:

1. an explicit `Final answer: X`;
2. `answer: X` / `answer is X`;
3. the last standalone letter A–E in the reply.

If no letter is found the answer is marked as `z`, so failed extractions stay visible.

### 4. LLM-as-a-judge — `src/judge.py`

Without labels, the same Gemma model is used as a judge with two yes/no questions per answer:

- **Grounding**: is the selected answer supported by the retrieved rules?
- **Context reliability**: do the retrieved rules contain the information needed to answer?

### 5. Export — `src/export.py`

Predictions are written to `outputs/exercise_predictions.csv` in the format required by Codabench.

## Results

| Metric                          | Value             |
| ------------------------------- | ----------------- |
| Codabench accuracy              | **0.760** (19/25) |
| Judge: grounding rate           | 76.0%             |
| Judge: context reliability rate | 96.0%             |
| Failed letter extractions       | 1/25              |

The accuracy went from 0.68 to 0.76 over successive iterations by:

- splitting the rules into sentence-based chunks;
- augmenting the query with numbers and keywords from the question and options;
- simplifying the prompt to ask only for the final letter;
- switching to deterministic (greedy) generation.

## Limitations

- **Small evaluation set.** With 25 questions, a single answer is worth 4 accuracy points.
- **The judge is not independent.** It is the same 1B model that generated the answers, so the judge
  metrics are a rough indication, not an accuracy estimate.
- **The grounding judge sees only the selected letter**, not the text of the options, which limits how
  well it can verify an answer.
- **Letter extraction can fail.** One question was answered with no recognizable letter (`z`).

### Notebook

Open `notebooks/footsquare.ipynb` in Colab and run the cells in order. The first cell clones the
repository so that `src/` is importable:

```python
!git clone https://github.com/Gabri03ele/footsquare-rag.git
%cd footsquare-rag
```

### Command line

From the repository root:

```bash
python -m src.run                 # full pipeline, including the judge
python -m src.run --skip-judge    # predictions only
python -m src.run --n 5           # retrieve a different number of rules
```

Options: `--data-dir`, `--cache-dir`, `--out` change the data folder, the model cache and the output CSV.

### Dependencies

`torch`, `transformers`, `accelerate`, `sentence-transformers`, `numpy`, `requests`.
They are preinstalled on Colab; elsewhere:

```bash
pip install torch transformers accelerate sentence-transformers numpy requests
```
