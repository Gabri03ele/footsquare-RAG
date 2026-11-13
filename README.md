# Footsquare RAG

A Retrieval-Augmented Generation (RAG) pipeline that answers multiple-choice questions about the rules
of **Footsquare**, an invented team sport. Because the sport does not exist, a language model cannot
answer from prior knowledge: every answer has to be grounded in the rulebook retrieved at query time.

**Result: 0.84 accuracy (21/25)**, up from 0.68 for the same pipeline with free-text generation.

Built for the Natural Language Processing course of the MSc in Data Science at the University of Padova.

---

## The task

Footsquare is played on a circular pitch with raised baskets instead of goals. Its rulebook covers the
pitch (central circle, control ring, quadrant lines), scoring (baskets, a 5 m scoring arc worth 1 or 2
points), keeper zones, players and substitutions, match duration, restarts, fouls, cards and video review.

- **Knowledge base**: 49 short rules (`rules.json`), one or two sentences each.
- **Questions**: 25 multiple-choice questions (`questions_mcq_student.json`) with options A–E;
  every question includes the distractor "There is not enough information".
- **Evaluation**: accuracy against hand-made gold labels (`eval/gold_labels.json`), written by reading
  the rulebook.

## Pipeline

For each question the pipeline runs five steps:

| Step        | What happens                                                                                                                                                                                 | Module            |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| 1. Index    | The rules are split into chunks and embedded once, before any question is asked.                                                                                                             | `src/index.py`    |
| 2. Retrieve | The question is enriched with numbers and keywords from its options, embedded, and compared with every chunk; the 7 most similar rules are kept.                                             | `src/retrieve.py` |
| 3. Answer   | The language model scores every option given the 7 rules, with the options shown in five different orders; its scores are combined with how closely each option matches the retrieved rules. | `src/generate.py` |
| 4. Judge    | The same model checks whether the answer is supported by the rules and whether the rules were relevant to the question.                                                                      | `src/judge.py`    |
| 5. Export   | The letters are saved in a CSV with the columns `id` and `prediction`.                                                                                                                       | `src/export.py`   |

The sections below describe each step in detail.

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

### 3. Answering — `src/generate.py`

The final method (`hybrid`) combines two scores for every option, with equal weight.

**Language model, averaged over option orders.** The retrieved rules, the question and the options are
given to [`unsloth/gemma-3-1B-it`](https://huggingface.co/unsloth/gemma-3-1B-it). Instead of letting the
model write an answer, the reply is started with `Final answer:` and the probability of the next token
being each letter is read directly from the model. This is repeated with the options rotated five times
(ABCDE, BCDEA, …), relabelled A–E each time, and the probabilities are mapped back to the original options
and averaged. The rotation removes a strong position bias: with the original order the model never chose
the last option, even when the only rule in its context stated it word for word.

**Option–rule similarity.** Each option is embedded together with its question, and its highest cosine
similarity with the 7 retrieved rules is turned into a probability with a softmax (temperature 0.05).

The weight (0.5) and the temperature were fixed before measuring the result. The other methods are still
available for comparison: `generate` (free-text answer with the letter extracted from the reply),
`logits` (letter probabilities in the original order only) and `permute` (language model only, averaged
over the rotations).

### 4. LLM-as-a-judge — `src/judge.py`

The same Gemma model is used as a judge with two yes/no questions per answer:

- **Grounding**: is the selected answer supported by the retrieved rules?
- **Context reliability**: do the retrieved rules contain the information needed to answer?

It was useful while no labels were available; with the gold labels it is only a secondary signal.

### 5. Export — `src/export.py`

Predictions are written to `outputs/exercise_predictions.csv` with the columns `id` and `prediction`.

## Results

Accuracy on the 25 questions, same retrieval (top 7 rules) for every method:

| Method                                                         | Accuracy         |
| -------------------------------------------------------------- | ---------------- |
| `generate` — free-text answer, letter extracted from the reply | 0.68 (17/25)     |
| `permute` — letter probabilities averaged over 5 option orders | 0.76 (19/25)     |
| **`hybrid` — `permute` + option–rule similarity (final)**      | **0.84 (21/25)** |
| Option–rule similarity only, without the language model        | 0.88 (22/25)     |

Findings:

- **Position bias.** With free-text generation and with letter probabilities in the original order,
  the model never picked option E. Averaging over rotated orders fixed four questions.
- **Retrieval is not the bottleneck.** In the questions answered wrongly by the language model the correct
  rule was almost always among those retrieved, often in first position.
- **A 1B model adds little here.** Matching the options against the retrieved rules alone scores one
  question more than the hybrid: the correct options are close paraphrases of the rules, which the
  embedding model captures better than the small language model. The hybrid is kept as the final method
  because it was chosen before seeing the results and keeps the language model in the loop.

## Limitations

- **Small evaluation set.** With 25 questions, a single answer is worth 4 accuracy points, and runs on
  the GPU can differ by one answer.
- **Tuned on the evaluation set.** The methods were compared on the same 25 questions used to report the
  accuracy, so the results are optimistic for unseen questions.
- **Hand-made labels.** The gold labels were written from the rulebook, not provided by the course.
- **The judge is not independent.** It is the same 1B model that answers, and the grounding judge sees
  only the selected letter, not the text of the option.

### Command line

From the repository root:

```bash
python -m src.run                              # final method (hybrid), including the judge
python -m src.run --skip-judge                 # predictions only
python -m src.run --method generate            # another answering method
python -m src.run --n 5                        # retrieve a different number of rules
```

### Dependencies

`torch`, `transformers`, `accelerate`, `sentence-transformers`, `numpy`, `requests`.
They are preinstalled on Colab; elsewhere:

```bash
pip install torch transformers accelerate sentence-transformers numpy requests
```
