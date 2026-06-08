from .base import BaseParser
from .clova_parser import ClovaOCRParser
from .pymupdf4llm_parser import PyMuPDF4LLMParser
from .pymupdf_parser import PyMuPDFParser

_REGISTRY: dict[str, type[BaseParser]] = {
    "pymupdf": PyMuPDFParser,
    "pymupdf4llm": PyMuPDF4LLMParser,
    "clova": ClovaOCRParser,
}


def get_parser(parser_type: str) -> BaseParser:
    if parser_type not in _REGISTRY:
        raise ValueError(
            f"지원하지 않는 파서: {parser_type}. 사용 가능: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[parser_type]()


def get_all_parsers() -> dict[str, BaseParser]:
    return {name: cls() for name, cls in _REGISTRY.items()}
