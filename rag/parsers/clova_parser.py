import base64
import os
import uuid

import requests

from .base import BaseParser


class ClovaOCRParser(BaseParser):
    _TIMEOUT = 30

    def __init__(self):
        self.invoke_url = os.getenv("CLOVA_OCR_INVOKE_URL")
        self.secret = os.getenv("CLOVA_OCR_SECRET")

        if not self.invoke_url or not self.secret:
            raise EnvironmentError(
                "CLOVA OCR 환경변수가 설정되지 않았습니다. "
                "CLOVA_OCR_INVOKE_URL, CLOVA_OCR_SECRET를 .env에 추가하세요."
            )

    def parse(self, file_path: str) -> str:
        ext = file_path.rsplit(".", 1)[-1].lower()

        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "images": [{"format": ext, "name": "document", "data": encoded}],
            "requestId": str(uuid.uuid4()),
            "version": "V2",
            "timestamp": 0,
        }
        headers = {
            "X-OCR-SECRET": self.secret,
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.invoke_url, headers=headers, json=payload, timeout=self._TIMEOUT
        )
        response.raise_for_status()

        fields = response.json()["images"][0].get("fields", [])
        return "\n".join(f["inferText"] for f in fields)
