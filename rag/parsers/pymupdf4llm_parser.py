import pymupdf4llm

from .base import BaseParser


class PyMuPDF4LLMParser(BaseParser):
    def parse(self, file_path: str) -> str:
        return pymupdf4llm.to_markdown(file_path)
