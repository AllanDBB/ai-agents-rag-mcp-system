"""
Procesador de documentos para el RAG.

Nombre de archivo esperado: {semana}_SEMANA_AI_{fecha}_{num}.pdf
Ejemplo: 3_SEMANA_AI_20260303_1.pdf
"""
import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

_FILENAME_RE = re.compile(
    r"^(\d+)_[Ss][Ee][Mm][Aa][Nn][Aa]_[Aa][Ii]?_(\d{8})_(\d+)",
    re.IGNORECASE,
)


def _extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


def _clean(text: str) -> str:
    # Reagrupar palabras divididas por guión al final de línea
    text = re.sub(r"([a-záéíóúüñ])-\s+([a-záéíóúüñ])", r"\1\2", text, flags=re.I)
    # Colapsar espacios múltiples
    text = re.sub(r"[ \t]+", " ", text)
    # Reducir saltos de línea excesivos
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_filename(pdf_path: str) -> dict:
    name = Path(pdf_path).stem
    m = _FILENAME_RE.match(name)
    if m:
        semana, fecha_raw, num = m.group(1), m.group(2), m.group(3)
        fecha = f"{fecha_raw[:4]}-{fecha_raw[4:6]}-{fecha_raw[6:]}"
        return {
            "source": Path(pdf_path).name,
            "file_path": str(pdf_path),
            "semana": int(semana),
            "fecha": fecha,
            "documento_num": int(num),
            "tema_principal": f"Semana {semana}",
            "clase_asociada": f"Semana {semana} — {fecha}",
        }
    return {
        "source": Path(pdf_path).name,
        "file_path": str(pdf_path),
        "semana": 0,
        "fecha": "",
        "documento_num": 0,
        "tema_principal": "Desconocido",
        "clase_asociada": "Desconocida",
    }


def process_config_a(pdf_path: str) -> list[dict]:
    """Config A — chunk fijo: 512 caracteres, solapamiento 64."""
    text = _clean(_extract_text(pdf_path))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.RAG_CHUNK_SIZE_A,
        chunk_overlap=config.RAG_CHUNK_OVERLAP_A,
    )
    chunks = splitter.split_text(text)
    meta = _parse_filename(pdf_path)
    return [
        {"text": c, "metadata": {**meta, "config": "A", "chunk_index": i}}
        for i, c in enumerate(chunks)
    ]


def process_config_b(pdf_path: str) -> list[dict]:
    """Config B — chunk semántico: 1024 caracteres, separadores naturales del texto."""
    text = _clean(_extract_text(pdf_path))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.RAG_CHUNK_SIZE_B,
        chunk_overlap=config.RAG_CHUNK_OVERLAP_B,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
    )
    chunks = splitter.split_text(text)
    meta = _parse_filename(pdf_path)
    return [
        {"text": c, "metadata": {**meta, "config": "B", "chunk_index": i}}
        for i, c in enumerate(chunks)
    ]
