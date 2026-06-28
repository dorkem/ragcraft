import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from preprocess.chunker import chunk_documents
from preprocess.embedder import ClovaStudioEmbeddings
from preprocess.parser import parse_pdf
from preprocess.vectorstore import add_documents, get_vectorstore

UPLOAD_DIR = Path("upload")


def run(pdf_paths: list[Path]) -> None:
    embeddings = ClovaStudioEmbeddings()
    vectorstore = get_vectorstore(embeddings)

    for pdf_path in pdf_paths:
        print(f"[파싱]  {pdf_path.name}")
        docs = parse_pdf(pdf_path)
        print(f"        → {len(docs)}페이지")

        print(f"[청킹]  {pdf_path.name}")
        chunks = chunk_documents(docs)
        print(f"        → {len(chunks)}청크 (chunk_size={800}, overlap={80})")

        print(f"[저장]  {pdf_path.name}")
        add_documents(vectorstore, chunks)
        print(f"        → 완료\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        targets = sorted(UPLOAD_DIR.glob("*.pdf"))

    if not targets:
        print(f"PDF 없음: {UPLOAD_DIR}")
        sys.exit(1)

    run(targets)
