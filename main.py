import subprocess
import sys
from pathlib import Path

UPLOAD_DIR = Path("upload")


def select_file() -> Path:
    pdf_files = sorted(UPLOAD_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"PDF 없음: {UPLOAD_DIR}")

    print("=== 전처리할 파일을 선택하세요 ===")
    for i, f in enumerate(pdf_files, 1):
        print(f"  [{i}] {f.name}")

    choice = int(input("\n번호 입력: "))
    return pdf_files[choice - 1]


if __name__ == "__main__":
    pdf_path = select_file()
    print(f"\n선택: {pdf_path.name}\n")
    subprocess.run([sys.executable, "preprocess.py", str(pdf_path)], check=True)
