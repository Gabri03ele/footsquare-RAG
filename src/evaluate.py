#Accuracy against hand-made gold labels (eval/gold_labels.json)

import json

def load_gold(path):
    #Return {question_id: correct_letter}
    gold = json.loads(path.read_text(encoding="utf-8"))
    return {qid: letter.strip().upper() for qid, letter in gold.items() if letter.strip()}

def accuracy(predictions, gold):
    errors = []
    for qid, correct in gold.items():
        predicted = predictions.get(qid)
        if predicted != correct:
            errors.append((qid, predicted, correct))
    total = len(gold)
    acc = (total - len(errors)) / total if total else 0.0
    return acc, errors

def print_report(predictions, gold, name="run"):
    acc, errors = accuracy(predictions, gold)
    correct = len(gold) - len(errors)
    print(f"{name}: {acc:.3f} ({correct}/{len(gold)})")
    for qid, predicted, right in errors:
        print(f"   {qid}: predicted {predicted}, correct {right}")