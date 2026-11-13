"""Answer generation with Gemma-3-1B-it from the retrieved rules."""

import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.retrieve import augment_query, retrieve_top_n

GENERATOR_ID = "unsloth/gemma-3-1B-it"

# Hybrid scoring, fixed before evaluation: equal weight to the language model and
# to the embedding similarity between each option and the retrieved rules.
HYBRID_WEIGHT = 0.5
SIM_TEMPERATURE = 0.05

SYSTEM_PROMPT = (
    "You are a Footsquare rules expert. You answer multiple choice questions by "
    "finding which option matches the provided rules. Always give one letter "
    "answer: {letters}."
)

USER_TEMPLATE = """Answer this question about the rules of Footsquare using ONLY the provided rules.

RULES:
{context}

QUESTION: {question}

OPTIONS:
{options}

INSTRUCTIONS:
- Read each option carefully
- Match the option that EXACTLY fits the rules
- Respond with ONLY: "Final answer: X" where X is {letters}

Final answer:"""


def load_generator(cache_dir):
    """Load tokenizer and model in float16, placed automatically on the available device."""
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_ID, cache_dir=str(cache_dir))
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_ID,
        cache_dir=str(cache_dir),
        dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, model


def format_options(question):
    """Return the options of a question as 'A) text' lines."""
    return "\n".join(f"{opt['id']}) {opt['text']}" for opt in question.get("options", []))


def format_letters(question):
    """Return the option letters as 'A, B, C, D, or E'."""
    letters = [opt["id"] for opt in question.get("options", [])]
    if len(letters) < 2:
        return "".join(letters)
    return ", ".join(letters[:-1]) + ", or " + letters[-1]


def build_context(chunks):
    """Join the retrieved chunks as numbered rules."""
    return "\n\n".join(f"[Rule {i + 1}] {chunk['text']}" for i, chunk in enumerate(chunks))


def build_messages(question_text, context, options_text, letters):
    """Build the chat messages for the generator."""
    user_message = USER_TEMPLATE.format(
        context=context, question=question_text, options=options_text, letters=letters
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(letters=letters)},
        {"role": "user", "content": user_message},
    ]


def _to_input_ids(tokenized, device):
    """apply_chat_template can return a tensor, a BatchEncoding or a dict depending
    on the transformers version: always return a tensor of input ids."""
    if isinstance(tokenized, torch.Tensor):
        return tokenized.to(device)
    if hasattr(tokenized, "input_ids"):
        return tokenized.input_ids.to(device)
    if isinstance(tokenized, dict) and "input_ids" in tokenized:
        return tokenized["input_ids"].to(device)
    return torch.tensor([tokenized]).to(device)


def generate_text(messages, tokenizer, model, max_new_tokens=50, max_length=4096):
    """Greedy-decode a reply to the messages and return only the generated text."""
    tokenized = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = _to_input_ids(tokenized, model.device)
    input_length = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()


def letter_scores(messages, letters, tokenizer, model):
    """Score each option letter by the probability of being the next token
    after 'Final answer:' at the start of the model's reply.

    Returns {letter: probability}, normalized over the given letters.
    """
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt += "Final answer:"
    # the chat template already contains the <bos> token
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        logits = model(input_ids).logits[0, -1].float()

    scores = {}
    for letter in letters:
        # a letter can be tokenized as "A" or " A": take the best of the two
        candidates = []
        for variant in (letter, " " + letter):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                candidates.append(logits[ids[0]])
        scores[letter] = torch.stack(candidates).max()

    probs = torch.softmax(torch.stack(list(scores.values())), dim=0)
    return {letter: float(p) for letter, p in zip(scores, probs)}


def extract_answer(text):
    """Extract the chosen letter from the model output; return 'z' if none is found."""
    match = re.search(r"[Ff]inal\s+[Aa]nswer[:\s]+([A-E])", text)
    if match:
        return match.group(1).upper()

    match = re.search(r"[Aa]nswer[:\s]+(?:is\s+)?([A-E])", text)
    if match:
        return match.group(1).upper()

    # fall back to the last standalone capital letter mentioned
    # (no .upper(): it would turn the article "a" into the option "A")
    matches = re.findall(r"\b([A-E])\b", text)
    if matches:
        return matches[-1]

    return "z"


def _rotated_question(question, shift):
    """Return a copy of the question with the options rotated by `shift` and relabelled
    A, B, C...; also return the map from new letter to original option id."""
    options = question["options"]
    rotated = options[shift:] + options[:shift]
    labels = [opt["id"] for opt in options]  # original label order, e.g. A..E
    new_options = [{"id": labels[i], "text": opt["text"]} for i, opt in enumerate(rotated)]
    to_original = {labels[i]: opt["id"] for i, opt in enumerate(rotated)}
    return {**question, "options": new_options}, to_original


def permutation_scores(question, context, tokenizer, model):
    """Average the letter probabilities over all cyclic orders of the options.

    Returns ({original_letter: probability}, [original option chosen in each rotation]).
    """
    original_ids = [opt["id"] for opt in question["options"]]
    totals = {letter: 0.0 for letter in original_ids}
    picks = []
    for shift in range(len(original_ids)):
        rotated, to_original = _rotated_question(question, shift)
        messages = build_messages(
            rotated["question"], context, format_options(rotated), format_letters(rotated)
        )
        scores = letter_scores(messages, list(to_original), tokenizer, model)
        for new_letter, prob in scores.items():
            totals[to_original[new_letter]] += prob / len(original_ids)
        picks.append(to_original[max(scores, key=scores.get)])
    return totals, picks


def option_similarity_scores(question, top_chunks, kb_chunks, embedder):
    """Score each option by its highest cosine similarity with the retrieved rules.

    Each option is embedded as "question + option text"; the similarities are turned
    into probabilities with a softmax at temperature SIM_TEMPERATURE.
    Returns ({letter: probability}, {letter: raw similarity}).
    """
    emb_by_id = {chunk["id"]: chunk["embedding"] for chunk in kb_chunks}
    rule_emb = np.array([emb_by_id[chunk["id"]] for chunk in top_chunks])

    letters = [opt["id"] for opt in question["options"]]
    texts = [f"{question['question']} {opt['text']}" for opt in question["options"]]
    option_emb = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    sims = (option_emb @ rule_emb.T).max(axis=1)
    weights = np.exp((sims - sims.max()) / SIM_TEMPERATURE)
    probs = weights / weights.sum()
    return (
        {letter: float(p) for letter, p in zip(letters, probs)},
        {letter: float(s) for letter, s in zip(letters, sims)},
    )


def answer_question(question, kb_chunks, embedder, tokenizer, model, n=7, method="hybrid"):
    """Run retrieval and generation for one question and return the full record.

    method="generate": the model writes its answer and the letter is extracted from the text.
    method="logits":   the letter with the highest next-token probability is chosen.
    method="permute":  the options are presented in every cyclic order; the letter
                       probabilities are mapped back to the original options and averaged.
    method="hybrid":   "permute" probabilities combined with the embedding similarity
                       between each option and the retrieved rules.
    """
    options_text = format_options(question)
    query = augment_query(question["question"], options_text)
    top_chunks = retrieve_top_n(query, kb_chunks, embedder, n=n)
    context = build_context(top_chunks)

    record = {
        "id": question["id"],
        "query": question["question"],
        "retrieved": top_chunks,
        "options": options_text,
        "reasoning": "",
    }

    if method == "generate":
        messages = build_messages(question["question"], context, options_text, format_letters(question))
        raw_output = generate_text(messages, tokenizer, model)
        record["answer"] = extract_answer(raw_output)
        record["reasoning"] = raw_output
        return record

    if method == "logits":
        messages = build_messages(question["question"], context, options_text, format_letters(question))
        letters = [opt["id"] for opt in question["options"]]
        scores = letter_scores(messages, letters, tokenizer, model)
        record["answer"] = max(scores, key=scores.get)
        record["scores"] = scores
        return record

    if method in ("permute", "hybrid"):
        llm_scores, picks = permutation_scores(question, context, tokenizer, model)
        record["picks"] = picks
        record["llm_scores"] = llm_scores

        if method == "permute":
            record["scores"] = llm_scores
        else:
            emb_scores, sims = option_similarity_scores(question, top_chunks, kb_chunks, embedder)
            record["emb_scores"] = emb_scores
            record["similarities"] = sims
            record["scores"] = {
                letter: HYBRID_WEIGHT * llm_scores[letter] + (1 - HYBRID_WEIGHT) * emb_scores[letter]
                for letter in llm_scores
            }

        record["answer"] = max(record["scores"], key=record["scores"].get)
        return record

    raise ValueError(f"Unknown method: {method}")