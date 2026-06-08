from pathlib import Path

from dotenv import load_dotenv

from core.logger import get_logger, setup_logging
from rag.converter import to_pdf
from rag.parsers.factory import PARSER_TYPE, get_parser
from rag.writer import ResultWriter

load_dotenv()
setup_logging()
logger = get_logger(__name__)


def select_file(directory: str) -> Path:
    dir_path = Path(directory)
    files = sorted(f for f in dir_path.iterdir() if f.is_file())

    if not files:
        raise FileNotFoundError(f"{directory} 폴더에 파일이 없습니다.")

    print("\n[파일 선택]")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f.name}")

    while True:
        raw = input("\n번호를 입력하세요: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(files):
            chosen = files[int(raw) - 1]
            logger.info("파일 선택: %s", chosen.name)
            return chosen
        print(f"  1 ~ {len(files)} 사이의 번호를 입력하세요.")


def run_pipeline(file_path: Path) -> str:
    logger.info("PDF 변환 시작: %s", file_path.name)
    pdf_path = to_pdf(str(file_path))
    logger.info("PDF 변환 완료: %s", pdf_path)

    logger.info("파서: %s", PARSER_TYPE)
    parser = get_parser()
    result = parser.parse(pdf_path)
    logger.info("파싱 완료 (%d자)", len(result))
    return result


def main():
    file_path = select_file("./test_sample")
    result = run_pipeline(file_path)

    writer = ResultWriter()
    out_file = writer.save(result, file_path, tag=PARSER_TYPE, stage="parsing")
    logger.info("결과 저장: %s", out_file)

    print("\n" + "-" * 40)
    print(result[:500])
    print(f"\n결과 파일: {out_file}")


if __name__ == "__main__":
    main()
