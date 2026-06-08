import os

from llama_parse import LlamaParse

from .base import BaseParser


class LlamaParseParser(BaseParser):
    def __init__(self):
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "LLAMA_CLOUD_API_KEY가 설정되지 않았습니다. "
                "https://cloud.llamaindex.ai 에서 발급 후 .env에 추가하세요."
            )
        self.parser = LlamaParse(api_key=api_key, result_type="markdown")

    def parse(self, file_path: str) -> str:
        documents = self.parser.load_data(file_path)
        return "\n\n".join(doc.text for doc in documents)
