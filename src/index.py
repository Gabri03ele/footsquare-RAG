
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBEDDER_ID = "Qwen/Qwen3-Embedding-0.6B"

def load_embedder(cache_dir: Path) -> SentenceTransformer:
    #Load the sentence embedding model, caching weights in cache_dir
    return SentenceTransformer(EMBEDDER_ID, cache_folder=str(cache_dir))

def split_chunks(text: str, max_length: int = 200, overlap: int = 40) -> list:
    #Split text on sentence boundaries into chunks of at most max_length words
    #Consecutive chunks share up to `overlap` words so that
    #context is not lost
    
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) > max_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            # keep the last sentences of the previous chunk as overlap
            overlap_sentences = []
            temp_length = 0
            for s in reversed(current_chunk):
                temp_length += len(s.split())
                if temp_length > overlap:
                    break
                overlap_sentences.insert(0, s)
            current_chunk = overlap_sentences
            current_length = temp_length

        current_chunk.append(sentence)
        current_length += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def build_index(
    rules: dict,
    embedder: SentenceTransformer,
    max_length: int = 200,
    overlap: int = 40,
) -> list:
    #Chunk every rule and embed each chunk with normalized embeddings
    #Returns a list of dicts with keys: id, text, embedding, parent_id.

    kb_chunks = []
    for rule_id, text in rules.items():
        for idx, chunk_text in enumerate(split_chunks(text, max_length, overlap)):
            embedding = embedder.encode(
                chunk_text, convert_to_numpy=True, normalize_embeddings=True
            )
            kb_chunks.append(
                {
                    "id": f"{rule_id}_chunk{idx}",
                    "text": chunk_text,
                    "embedding": embedding,
                    "parent_id": rule_id,
                }
            )
    return kb_chunks