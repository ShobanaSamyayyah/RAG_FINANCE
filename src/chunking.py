"""Stage 2 (chunking) — fixed/no-overlap, fixed+overlap, and recursive strategies."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_fixed(pages: list[dict], chunk_size: int = 500, overlap: int = 0) -> list[dict]:
    """Naive fixed-width character windows (baseline, worst case: can cut mid-sentence)."""
    chunks = []
    for page in pages:
        text = page["text"]
        step = max(chunk_size - overlap, 1)
        for start in range(0, len(text), step):
            piece = text[start:start + chunk_size].strip()
            if piece:
                chunks.append({"text": piece, "source": page["source"], "page": page["page"]})
    return chunks


def chunk_recursive(pages: list[dict], chunk_size: int = 500, overlap: int = 80) -> list[dict]:
    """Recursive/sentence-aware splitting — tries paragraph, then sentence, then word boundaries."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            if piece.strip():
                chunks.append({"text": piece, "source": page["source"], "page": page["page"]})
    return chunks


STRATEGIES = {
    "fixed_no_overlap": lambda pages, size: chunk_fixed(pages, size, overlap=0),
    "fixed_overlap": lambda pages, size: chunk_fixed(pages, size, overlap=min(100, size // 5)),
    "recursive": lambda pages, size: chunk_recursive(pages, size, overlap=min(100, size // 5)),
}
