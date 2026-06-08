from docling.document_converter import DocumentConverter

from .base import BaseParser


class DoclingParser(BaseParser):
    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, file_path: str) -> str:
        result = self.converter.convert(file_path)
        return result.document.export_to_markdown()
