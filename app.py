import subprocess
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="기술지침 RAG", page_icon="📋", layout="wide")

# ── 세션 초기화 ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None


@st.cache_resource(show_spinner="RAG 체인 로딩 중...")
def load_chain():
    from rag.chains.rag_chain import build_chain
    return build_chain()


def run_step(module: str, label: str, args: list = None) -> bool:
    cmd = [sys.executable, "-m", f"pipeline.{module}"] + (args or [])
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.stdout:
        st.code(result.stdout, language=None)
    if result.returncode != 0:
        st.error(f"❌ {label} 실패\n{result.stderr}")
        return False
    return True


# ── 사이드바 ────────────────────────────────────────────
with st.sidebar:
    st.header("📁 문서 관리")

    img_dir = Path("img")
    img_dir.mkdir(exist_ok=True)

    SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".hwp"}
    doc_files = sorted(
        f for f in img_dir.rglob("*")
        if f.is_file()
        and f.suffix.lower() in SUPPORTED
        and "complete_info" not in f.parts
    )

    if doc_files:
        st.write(f"**{len(doc_files)}개 문서**")
        for f in doc_files:
            st.caption(f"📄 {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
    else:
        st.info("img/ 폴더에 문서를 넣어주세요")

    processed_log = img_dir / "complete_info" / "processed.json"
    if processed_log.exists():
        import json
        log = json.loads(processed_log.read_text(encoding="utf-8"))
        done = len(log.get("processed", {}))
        st.caption(f"✅ 처리 완료: {done}개")

    st.divider()

    run_btn = st.button("🔄 파이프라인 실행", use_container_width=True, type="primary")
    if run_btn:
        steps = [
            ("step0_scan",  "스캔 & 중복 필터"),
            ("step1_parse", "문서 파싱"),
            ("step2_chunk", "청킹"),
            ("step3_embed", "임베딩"),
            ("step4_store", "ChromaDB 저장"),
        ]
        with st.status("파이프라인 실행 중...", expanded=True) as status:
            for module, label in steps:
                st.write(f"**{label}**")
                ok = run_step(module, label)
                if not ok:
                    status.update(label=f"{label} 실패", state="error")
                    break
            else:
                status.update(label="완료 ✅", state="complete")
                st.cache_resource.clear()  # 체인 캐시 초기화 (새 문서 반영)

    st.divider()

    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── 메인 ────────────────────────────────────────────────
st.title("📋 기술지침 RAG")
st.caption(f"모델: {__import__('os').getenv('HYPERCLOVA_MODEL', 'HCX-DASH-002')} | 임베딩: BGE-M3 | 벡터 DB: ChromaDB")

# 채팅 히스토리 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 질문 입력
if query := st.chat_input("기술 지침에 대해 질문하세요"):
    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # RAG 답변
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            try:
                from langchain_core.callbacks import BaseCallbackHandler

                class _TokenCapture(BaseCallbackHandler):
                    usage: dict = {}
                    def on_llm_end(self, response, **kwargs):
                        if response.llm_output:
                            self.usage = response.llm_output.get("token_usage", {})

                cb = _TokenCapture()
                chain = load_chain()
                answer = chain.invoke(query, config={"callbacks": [cb]})
                token_usage = cb.usage
            except Exception as e:
                answer = f"오류가 발생했습니다: {e}\n\n파이프라인을 먼저 실행해주세요."
                token_usage = {}

        st.markdown(answer)
        if token_usage:
            st.caption(
                f"입력 {token_usage.get('input_tokens', 0):,}토큰 / "
                f"출력 {token_usage.get('output_tokens', 0):,}토큰 / "
                f"합계 {token_usage.get('total_tokens', 0):,}토큰"
            )
        st.session_state.messages.append({"role": "assistant", "content": answer})
