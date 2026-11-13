# Answer generation with Gemma-3-1B-it from the retrieved rules
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.retrieve import augment_query, retrieve_top_n

GENERATOR_ID = "unsloth/gemma-3-1B-it"

# the knowledge base is about Footsquare.
SYSTEM_PROMPT = (
    "You are a cricket rules expert. You answer multiple choice questions by "
    "finding which option matches the provided rules. Always give one letter "
    "answer: A, B, C, D, or E."
)

USER_TEMPLATE = """Answer this cricket rules question using ONLY the provided rules.

        RULES:
        {context}

        QUESTION: {question}

        OPTIONS:
        {options}

        INSTRUCTIONS:
        - Read each option carefully
        - Match the option that EXACTLY fits the rules
        - Respond with ONLY: "Final answer: X" where X is A, B, C, D, or E

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


def build_context(chunks):
    """Join the retrieved chunks as numbered rules."""
    return "\n\n".join(f"[Rule {i + 1}] {chunk['text']}" for i, chunk in enumerate(chunks))


def build_messages(question_text, context, options_text):
    """Build the chat messages for the generator."""
    user_message = USER_TEMPLATE.format(
        context=context, question=question_text, options=options_text
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
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


def extract_answer(text):
    """Extract the chosen letter from the model output; return 'z' if none is found."""
    match = re.search(r"[Ff]inal\s+[Aa]nswer[:\s]+([A-E])", text)
    if match:
        return match.group(1).upper()

    match = re.search(r"[Aa]nswer[:\s]+(?:is\s+)?([A-E])", text)
    if match:
        return match.group(1).upper()

    # fall back to the last standalone letter mentioned
    matches = re.findall(r"\b([A-E])\b", text.upper())
    if matches:
        return matches[-1]

    return "z"


def answer_question(question, kb_chunks, embedder, tokenizer, model, n=7):
    """Run retrieval and generation for one question and return the full record."""
    options_text = format_options(question)
    query = augment_query(question["question"], options_text)
    top_chunks = retrieve_top_n(query, kb_chunks, embedder, n=n)

    messages = build_messages(question["question"], build_context(top_chunks), options_text)
    raw_output = generate_text(messages, tokenizer, model)

    return {
        "id": question["id"],
        "query": question["question"],
        "retrieved": top_chunks,
        "answer": extract_answer(raw_output),
        "reasoning": raw_output,
        "options": options_text,
    }