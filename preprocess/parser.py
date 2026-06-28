import re
from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document

_PICTURE_BLOCK_RE = re.compile(
    r"----- Start of picture text -----.*?----- End of picture text -----\n*",
    re.DOTALL,
)
_PICTURE_TAG_RE = re.compile(r"==> picture \[.*?\] <==")
_SEPARATOR_LINE_RE = re.compile(r"[━─═\-]{4,}")


def _clean(text: str) -> str:
    text = _PICTURE_BLOCK_RE.sub(" ", text)
    text = _PICTURE_TAG_RE.sub(" ", text)
    text = _SEPARATOR_LINE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_pdf(pdf_path: Path) -> list[Document]:
    pages = pymupdf4llm.to_text(str(pdf_path), page_chunks=True)
    docs = []
    for page in pages:
        metadata = page.get("metadata", {})
        docs.append(Document(
            page_content=_clean(page["text"]),
            metadata={
                "source": pdf_path.name,
                "page_number": metadata.get("page_number", metadata.get("page", 0) + 1),
                "total_pages": metadata.get("page_count", len(pages)),
                "file_path": str(pdf_path),
            },
        ))
    return docs
