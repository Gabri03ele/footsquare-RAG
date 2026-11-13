# Download and load the Footsquare knowledge base and the MCQ

import json
from pathlib import Path
import requests

BASE_URL = ("https://raw.githubusercontent.com/ferrazzipietro/dataset-nlp-math/"
            "main/data_rag/excercise")
RULES_FILE = "rules.json"
QUESTIONS_FILE = "questions_mcq_student.json"

def download_data(data_dir: Path) -> None:
    # Download rules and questions into data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in (RULES_FILE, QUESTIONS_FILE):
        path = data_dir / name
        if path.exists():
            continue
        
        response = requests.get(f"{BASE_URL}/{name}", timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
        print(f"Downloaded {name} to {path}")

def load_rules(data_dir: Path) -> dict:
    #Return the knowledge base as {rule_id: rule_text}
    return json.loads((data_dir / RULES_FILE).read_text(encoding="utf-8"))


def load_questions(data_dir: Path) -> list:
    # Return the list of questions, each with id, question and options
    return json.loads((data_dir / QUESTIONS_FILE).read_text(encoding="utf-8"))