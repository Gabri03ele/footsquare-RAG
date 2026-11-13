import re
import numpy as np

# Keywords appended to the query when they appear in the question or options
# The list was originally written with cricket in mind; several terms (ball,
# boundary, run, over...) also occur in the Footsquare rules. It is kept as in
# the version evaluated on Codabench (0.760).
KEYWORD_PATTERN = (
    r"\b(?:wicket|over|ball|run|boundary|innings|striker|non-striker|"
    r"bowler|fielder|umpire|dead ball|no ball|wide|bye|leg bye)\w*\b"
)

def augment_query(question, options_text):
    #Append numbers and keywords found in the question and options to the question
    text = question + " " + options_text
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    keywords = re.findall(KEYWORD_PATTERN, text, re.IGNORECASE)
    # dict.fromkeys removes duplicates while keeping a fixed order
    extra_terms = list(dict.fromkeys(numbers + keywords))
    return f"{question} {' '.join(extra_terms)}"


def embed_query(query, embedder):
    #normalized embedding of a single query
    return embedder.encode(query, convert_to_numpy=True, normalize_embeddings=True)


def cosine_similarity_matrix(query_emb, kb_emb):
    #Return a 1 x N matrix of cosine similarities.
    #Both the query and the knowledge-base embeddings are normalized, so the
    #cosine similarity is the dot product
    
    similarities = np.dot(kb_emb, query_emb)
    return similarities.reshape(1, -1)


def retrieve_top_n(query, kb_chunks, embedder, n=7):
    #Return the n chunks most similar to the query
    query_emb = embed_query(query, embedder)
    kb_emb = np.array([chunk["embedding"] for chunk in kb_chunks])
    similarities = cosine_similarity_matrix(query_emb, kb_emb).flatten()

    top_indices = np.argsort(similarities)[::-1][:n]

    results = []
    for idx in top_indices:
        results.append(
            {
                "id": kb_chunks[idx]["id"],
                "text": kb_chunks[idx]["text"],
                "similarity": float(similarities[idx]),
            }
        )
    return results