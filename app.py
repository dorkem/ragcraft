from pathlib import Path

from dotenv import load_dotenv

from core.logger import get_logger, setup_logging
from rag.converter import to_pdf
from rag.parsers.factory import get_all_parsers
from rag.writer import ResultWriter

load_dotenv()
setup_logging()
logger = get_logger(__name__)


def run_all(sample_dir: str = "./test_sample") -> None:
    dir_path = Path(sample_dir)
    files = sorted(f for f in dir_path.iterdir() if f.is_file())

    if not files:
        raise FileNotFoundError(f"{sample_dir} 폴더에 파일이 없습니다.")

    parsers = get_all_parsers()
    writer = ResultWriter()

    for file_path in files:
        logger.info("처리 시작: %s", file_path.name)

        try:
            pdf_path = to_pdf(str(file_path))
        except Exception as e:
            logger.error("PDF 변환 실패: %s - %s", file_path.name, e)
            continue

        for parser_name, parser in parsers.items():
            try:
                result = parser.parse(pdf_path)
                out_file = writer.save(
                    result, file_path, tag=parser_name, stage="parsing"
                )
                logger.info(
                    "[%s] 완료 → %s (%d자)", parser_name, out_file.name, len(result)
                )
            except Exception as e:
                logger.error("[%s] 파싱 실패: %s - %s", parser_name, file_path.name, e)


if __name__ == "__main__":
    run_all()
