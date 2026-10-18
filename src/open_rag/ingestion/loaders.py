"""Document loading: raw file -> normalized text + metadata.

WHY: parsing quality caps the quality of the whole system ("garbage in, garbage out").
Here we ship light loaders (md/txt/html/pdf). For complex PDFs (tables, scans,
multi-column) swap `load_pdf` for Docling / Unstructured behind the same interface.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path


@dataclass
class RawDocument:
    source: str
    text: str
    tenant: str = "default"
    metadata: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class _TextExtractor(HTMLParser):
    """Converts HTML to markdown-ish text, keeping headings so structure-aware chunking works."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "footer"}:
            self._skip += 1
        elif tag in {"h1", "h2", "h3", "h4"}:
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in {"p", "li", "br", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "footer"}:
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def load_html(path: Path) -> str:
    parser = _TextExtractor()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return "".join(parser.parts)


def load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


_LOADERS = {
    ".md": lambda p: p.read_text(encoding="utf-8"),
    ".txt": lambda p: p.read_text(encoding="utf-8"),
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
}

SUPPORTED_EXTENSIONS = tuple(_LOADERS)


def load_file(path: Path, tenant: str = "default") -> RawDocument:
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return RawDocument(source=path.name, text=loader(path), tenant=tenant)


def iter_corpus(root: Path, tenant: str = "default"):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _LOADERS:
            yield load_file(path, tenant)
