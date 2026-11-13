# Write predictions in the CSV format (id, prediction)

import csv
from collections import Counter

VALID_LETTERS = ("A", "B", "C", "D", "E")


def write_predictions(results, path, fallback=None):
    #Write one row per question and return the rows written.
    # Answers that are not a valid letter are
    # written as they are, unless `fallback` is given

    rows = []
    for result in results:
        prediction = result["answer"].strip()
        if prediction not in VALID_LETTERS:
            print(f"Warning: {result['id']} has invalid prediction '{prediction}'")
            if fallback is not None:
                prediction = fallback
        rows.append({"id": result["id"], "prediction": prediction})

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prediction"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def prediction_distribution(rows):
    #Count how many times each letter was predicted
    return Counter(row["prediction"] for row in rows)