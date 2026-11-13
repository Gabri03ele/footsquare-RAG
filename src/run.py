# Run the full pipeline from the command line.

# Usage, from the repository root:
#     python -m src.run
#     python -m src.run --n 7 --skip-judge


import argparse
from pathlib import Path

from src.data import download_data, load_questions, load_rules
from src.export import prediction_distribution, write_predictions
from src.generate import answer_question, load_generator
from src.index import build_index, load_embedder
from src.judge import compute_scores, judge_result

ROOT = Path(__file__).resolve().parent.parent

def parse_args():
    parser = argparse.ArgumentParser(description="Footsquare RAG pipeline")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "models_cache")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "exercise_predictions.csv")
    parser.add_argument("--n", type=int, default=7, help="number of chunks to retrieve")
    parser.add_argument("--skip-judge", action="store_true", help="do not run the LLM judge")
    return parser.parse_args()


def main():
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    download_data(args.data_dir)
    rules = load_rules(args.data_dir)
    questions = load_questions(args.data_dir)
    print(f"Rules: {len(rules)}  Questions: {len(questions)}")

    embedder = load_embedder(args.cache_dir)
    kb_chunks = build_index(rules, embedder)
    print(f"KB chunks: {len(kb_chunks)}")

    tokenizer, model = load_generator(args.cache_dir)
    results = []
    for q in questions:
        result = answer_question(q, kb_chunks, embedder, tokenizer, model, n=args.n)
        results.append(result)
        print(f"{result['id']}: {result['answer']}")

    rows = write_predictions(results, args.out)
    print(f"Predictions written to {args.out}")
    print("Distribution:", dict(prediction_distribution(rows)))

    if not args.skip_judge:
        judgments = [judge_result(r, tokenizer, model) for r in results]
        scores = compute_scores(judgments)
        print(f"Grounding rate:           {scores['grounding_rate']:.1%}")
        print(f"Context reliability rate: {scores['reliability_rate']:.1%}")
        print(f"Both good:                {scores['both_good_rate']:.1%}")


if __name__ == "__main__":
    main()