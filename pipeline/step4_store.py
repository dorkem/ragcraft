"""
STEP 4: ChromaDB 저장 (LangChain Chroma 래퍼)
입력: result/yyyy/mm/dd/embed/{파일명}.json
출력: ChromaDB 컬렉션 (settings.CHROMA_PERSIST_DIR)

날짜 직접 지정: python -m pipeline.step4_store 2025/06/18
"""
import json
import sys
from datetime import datetime
from pathlib import Path

TODAY = datetime.now().strftime("%Y/%m/%d")


def main():
    date = sys.argv[1] if len(sys.argv) >= 2 else TODAY
    embed_dir = Path(f"result/{date}/embed")

    if not embed_dir.exists():
        print(f"[ERROR] 임베딩 결과 없음: {embed_dir}")
        print("  step3_embed.py 를 먼저 실행하거나 날짜를 확인하세요")
        sys.exit(1)

    embed_files = sorted(embed_dir.glob("*.json"))
    if not embed_files:
        print(f"[INFO] {embed_dir} 에 임베딩 결과가 없습니다")
        sys.exit(0)

    from core.config import settings
    from langchain_chroma import Chroma
    from rag.embeddings.clova_embedding import ClovaEmbedding

    vectorstore = Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=ClovaEmbedding(),
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )

    print(f"\n[STEP 4] ChromaDB 저장")
    print(f"컬렉션: {settings.CHROMA_COLLECTION_NAME}")
    print(f"경로  : {settings.CHROMA_PERSIST_DIR}")

    total = 0
    for embed_file in embed_files:
        embedded = json.loads(embed_file.read_text(encoding="utf-8"))
        if not embedded:
            continue

        texts = [item["text"] for item in embedded]
        embeddings = [item["embedding"] for item in embedded]
        metadatas = [item["metadata"] for item in embedded]
        ids = [f"{item['metadata'].get('filename', 'doc')}_{item['chunk_id']}" for item in embedded]

        # langchain_chroma의 add_texts는 embeddings 파라미터를 무시하므로
        # 이미 계산된 벡터를 그대로 저장하기 위해 컬렉션에 직접 추가
        vectorstore._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total += len(embedded)
        print(f"  {embed_file.name} - {len(embedded)}개 청크 저장")

    count = vectorstore._collection.count()
    print(f"\n[완료] 이번 배치: {total}개 / 컬렉션 누적: {count}개")


if __name__ == "__main__":
    main()
