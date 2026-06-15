"""Document loaders (Box 1, RAG) — ingest real filings into RAG records.

Reads `.txt` / `.md` / `.pdf` from a file or directory into the
`{id, source, text}` record shape that `build_index` consumes. Drop real 10-Ks /
earnings-call transcripts into a folder and index them directly.
"""

import os


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader

    return "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)


def load_documents(path: str, source_prefix: str = "") -> list[dict]:
    """Load .txt/.md/.pdf from a file or directory into RAG records.

    Args:
        path: a file or a directory of documents.
        source_prefix: optional prefix for the citation `source` (e.g. a ticker).

    Returns:
        A list of {id, source, text} records (skips unsupported extensions).
    """
    files = (
        [path]
        if os.path.isfile(path)
        else [os.path.join(path, f) for f in sorted(os.listdir(path))]
    )
    records = []
    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        if ext in (".txt", ".md"):
            with open(fp, encoding="utf-8") as f:
                text = f.read()
        elif ext == ".pdf":
            text = _read_pdf(fp)
        else:
            continue
        name = os.path.basename(fp)
        text = text.strip()
        if text:
            records.append({"id": name, "source": f"{source_prefix}{name}", "text": text})
    return records
