# LLM-as-a-judge: estimate grounding and context reliability without labels

from src.generate import build_context, generate_text

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating RAG system outputs. Answer ONLY with 'Yes' or 'No'. No explanation."
)

GROUNDING_TEMPLATE = """Is the selected answer clearly supported by the context?

        CONTEXT:
        {context}

        QUESTION:
        {question}

        SELECTED ANSWER: {answer}

        Is "{answer}" supported by the rules? Yes or No:"""

RELIABILITY_TEMPLATE = """Does the context contain information to answer this question?

        CONTEXT:
        {context}

        QUESTION:
        {question}

        Does the context have the needed information? Yes or No:"""


def run_judge(prompt, tokenizer, model):
    """Ask the judge a yes/no question; anything that does not start with 'yes' counts as 'No'."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt + "\n\nYour answer (Yes or No only):"},
    ]
    reply = generate_text(messages, tokenizer, model, max_new_tokens=3, max_length=2048)
    return "Yes" if reply.lower().startswith("yes") else "No"


def judge_result(result, tokenizer, model):
    """Judge one record produced by answer_question."""
    if result["answer"] == "z":
        # no answer was extracted: nothing to judge
        return {
            "id": result["id"],
            "query": result["query"],
            "answer": result["answer"],
            "grounding": "No",
            "context_reliability": "No",
        }

    context = build_context(result["retrieved"])
    grounding_prompt = GROUNDING_TEMPLATE.format(
        context=context, question=result["query"], answer=result["answer"]
    )
    reliability_prompt = RELIABILITY_TEMPLATE.format(
        context=context, question=result["query"]
    )

    return {
        "id": result["id"],
        "query": result["query"],
        "answer": result["answer"],
        "grounding": run_judge(grounding_prompt, tokenizer, model),
        "context_reliability": run_judge(reliability_prompt, tokenizer, model),
    }


def compute_scores(judgments):
    """Return the share of answers judged grounded, reliable and both."""
    total = len(judgments)
    if total == 0:
        return {"grounding_rate": 0, "reliability_rate": 0, "both_good_rate": 0, "total_queries": 0}

    grounded = sum(j["grounding"] == "Yes" for j in judgments)
    reliable = sum(j["context_reliability"] == "Yes" for j in judgments)
    both = sum(j["grounding"] == "Yes" and j["context_reliability"] == "Yes" for j in judgments)

    return {
        "grounding_rate": grounded / total,
        "reliability_rate": reliable / total,
        "both_good_rate": both / total,
        "total_queries": total,
    }