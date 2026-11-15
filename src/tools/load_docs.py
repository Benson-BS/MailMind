DOCS_DIR = 'data/docs'



def load_docs() -> list[dict]:
    """The Cardinal Energy documents, sorted by ``doc_id``.

    Each document is a dict with three keys. ``doc_id`` is the file name
    without ``.md``. ``title`` is the text of the first line that starts with
    "# ", or the ``doc_id`` if there is no such line. ``text`` is the whole
    file. Line numbers in this assignment are 1-based positions in
    ``text.splitlines()``.
    """
    docs = []
    for path in sorted(DOCS_DIR.glob("*.md"), key=lambda p: p.stem):
        text = path.read_text(encoding="utf-8")
        title = next(
            (line[2:].strip() for line in text.splitlines() if line.startswith("# ")),
            path.stem,
        )
        docs.append({"doc_id": path.stem, "title": title, "text": text})
    return docs