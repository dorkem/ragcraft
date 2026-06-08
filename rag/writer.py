from datetime import datetime
from pathlib import Path


class ResultWriter:
    def __init__(self, base_dir: str = "result"):
        self.base_dir = Path(base_dir)

    def save(self, result: str, source_file: Path, tag: str, stage: str) -> Path:
        today = datetime.now().strftime("%Y%m%d")
        out_dir = self.base_dir / stage / today
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / f"{source_file.stem}_{tag}.txt"
        out_file.write_text(result, encoding="utf-8")
        return out_file
