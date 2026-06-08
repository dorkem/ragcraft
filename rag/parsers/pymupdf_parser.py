import fitz

from .base import BaseParser


class PyMuPDFParser(BaseParser):
    def parse(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)
