from pathlib import Path

_PDF = {".pdf"}
_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
_OFFICE = {".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls"}
_HWP = {".hwp", ".hwpx"}


def to_pdf(file_path: str, output_dir: str | None = None) -> str:
    """파일을 PDF로 변환하여 경로를 반환합니다. 이미 PDF면 그대로 반환합니다."""
    src = Path(file_path)
    ext = src.suffix.lower()

    if ext in _PDF:
        return str(src)

    out_dir = Path(output_dir) if output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (src.stem + ".pdf")

    if ext in _IMAGE:
        _image_to_pdf(src, dest)
    elif ext in _OFFICE:
        _office_to_pdf(src, dest)
    elif ext in _HWP:
        raise NotImplementedError(f"HWP 변환은 아직 지원되지 않습니다: {src.name}")
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")

    return str(dest)


def _image_to_pdf(src: Path, dest: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("pip install Pillow")

    Image.open(src).convert("RGB").save(dest, "PDF")


def _office_to_pdf(src: Path, dest: Path) -> None:
    try:
        from docx2pdf import convert
    except ImportError:
        raise ImportError("pip install docx2pdf")

    convert(str(src), str(dest))
