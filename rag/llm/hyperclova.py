from typing import Any, Iterator, List, Optional
import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field
from core.config import settings


def _to_clova_messages(messages: List[BaseMessage]) -> List[dict]:
    role_map = {
        SystemMessage: "system",
        HumanMessage: "user",
        AIMessage: "assistant",
    }
    return [
        {"role": role_map.get(type(m), "user"), "content": m.content}
        for m in messages
    ]


class HyperClovaLLM(BaseChatModel):
    """HyperCLOVA X (CLOVA Studio Chat Completions) LangChain 래퍼"""

    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.5)
    top_p: float = Field(default=0.8)
    repeat_penalty: float = Field(default=1.1)

    @property
    def _llm_type(self) -> str:
        return "hyperclova"

    def _call_api(self, messages: List[BaseMessage]) -> dict:
        url = f"{settings.CLOVA_STUDIO_ENDPOINT}/testapp/v1/chat-completions/{settings.HYPERCLOVA_MODEL}"
        payload = {
            "messages": _to_clova_messages(messages),
            "maxTokens": self.max_tokens,
            "temperature": self.temperature,
            "topP": self.top_p,
            "repeatPenalty": self.repeat_penalty,
            "stopBefore": [],
            "includeAiFilters": False,
        }
        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.CLOVA_STUDIO_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if data["status"]["code"] != "20000":
            raise RuntimeError(f"HyperCLOVA API 오류: {data['status']}")
        return data["result"]

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = self._call_api(messages)
        content = result["message"]["content"]
        token_usage = {
            "input_tokens": result.get("inputLength", 0),
            "output_tokens": result.get("outputLength", 0),
            "total_tokens": result.get("inputLength", 0) + result.get("outputLength", 0),
        }
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))],
            llm_output={"token_usage": token_usage, "model": settings.HYPERCLOVA_MODEL},
        )
