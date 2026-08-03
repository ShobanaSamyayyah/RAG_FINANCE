"""
Stage 1 (parsing) — 3 alternative PDF parsers behind one interface, plus a
CSV-to-narrative-chunks converter for FRED_Total_Consumer_Credit.csv.

Each parser returns the same shape: list[{"text", "source", "page"}], one
record per page, so downstream chunking/embedding is identical regardless
of which parser produced it.
"""

from pathlib import Path

import pandas as pd


def parse_with_pypdf(path: str) -> list[dict]:
    import pypdf

    path = Path(path)
    reader = pypdf.PdfReader(str(path))
    return [
        {"text": page.extract_text() or "", "source": path.name, "page": i + 1}
        for i, page in enumerate(reader.pages)
    ]


def parse_with_pdfplumber(path: str) -> list[dict]:
    import pdfplumber

    path = Path(path)
    records = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            records.append({"text": page.extract_text() or "", "source": path.name, "page": i + 1})
    return records


def parse_with_pymupdf(path: str) -> list[dict]:
    import fitz  # PyMuPDF

    path = Path(path)
    doc = fitz.open(str(path))
    records = [
        {"text": page.get_text(), "source": path.name, "page": i + 1}
        for i, page in enumerate(doc)
    ]
    doc.close()
    return records


PARSERS = {
    "pypdf": parse_with_pypdf,
    "pdfplumber": parse_with_pdfplumber,
    "pymupdf": parse_with_pymupdf,
}


def clean_text_ratio(pages: list[dict]) -> float:
    """% of pages whose extracted text looks intact (no empty/garbled pages)."""
    if not pages:
        return 0.0
    ok = 0
    for p in pages:
        text = p["text"]
        if not text or len(text.strip()) < 20:
            continue
        alpha = sum(c.isalpha() or c.isspace() for c in text)
        if alpha / max(len(text), 1) > 0.85:
            ok += 1
    return ok / len(pages)


def csv_to_narrative_chunks(path: str, date_col: str = "observation_date", value_col: str = "TOTALSL") -> list[dict]:
    """
    Turn the FRED total-consumer-credit time series into a handful of
    citable narrative chunks (not raw rows), so it flows through the same
    chunk -> embed -> retrieve -> cite pipeline as the PDFs. Values are in
    millions of dollars (FRED TOTALSL units).
    """
    path = Path(path)
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    latest = df.iloc[-1]
    year_ago_idx = df[date_col] <= (latest[date_col] - pd.DateOffset(months=12))
    year_ago = df[year_ago_idx].iloc[-1] if year_ago_idx.any() else df.iloc[0]
    pct_change_1y = (latest[value_col] - year_ago[value_col]) / year_ago[value_col] * 100

    peak = df.loc[df[value_col].idxmax()]
    trough = df.loc[df[value_col].idxmin()]

    decade_rows = df[df[date_col].dt.month == 1].copy()
    decade_rows["decade"] = (decade_rows[date_col].dt.year // 10) * 10
    decade_summary = decade_rows.groupby("decade")[value_col].last()

    chunks = []
    chunks.append({
        "text": (
            f"Total US consumer credit outstanding (FRED series TOTALSL) stood at "
            f"${latest[value_col]:,.0f} million as of {latest[date_col].date()}, compared with "
            f"${year_ago[value_col]:,.0f} million a year earlier ({year_ago[date_col].date()}). "
            f"That is a change of {pct_change_1y:+.1f}% over the trailing 12 months, meaning total "
            f"consumer credit has {'increased' if pct_change_1y >= 0 else 'decreased'} over the last year."
        ),
        "source": path.name, "page": 1,
    })
    chunks.append({
        "text": (
            f"Across the full FRED history of this series ({df[date_col].min().date()} to "
            f"{df[date_col].max().date()}), total US consumer credit reached its highest recorded "
            f"level of ${peak[value_col]:,.0f} million in {peak[date_col].date()}, and its lowest "
            f"recorded level of ${trough[value_col]:,.0f} million in {trough[date_col].date()}."
        ),
        "source": path.name, "page": 1,
    })
    decade_lines = "; ".join(f"{int(d)}s: ${v:,.0f} million" for d, v in decade_summary.items())
    chunks.append({
        "text": f"Total US consumer credit outstanding at the start of each decade (FRED TOTALSL): {decade_lines}.",
        "source": path.name, "page": 1,
    })
    return chunks
