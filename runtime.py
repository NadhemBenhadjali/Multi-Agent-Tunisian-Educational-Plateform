# -*- coding: utf-8 -*-
# config_files/api/runtime.py
from __future__ import annotations
from pathlib import Path
import os

# ── Windows OCR safety: point pytesseract to your local tessdata BEFORE other imports
try:
    import pytesseract, pathlib
    if os.name == "nt":
        # project-local tessdata (adjust if you keep it elsewhere)
        os.environ.setdefault(
            "TESSDATA_PREFIX",
            str((pathlib.Path(__file__).resolve().parents[1] / "tessdata").resolve())
        )
        # system tesseract.exe (adjust if installed elsewhere)
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    pass

from retrieval import build_retriever, ChapterRetrieverTool
from agents import build_llm, define_agents
from pdf_report import SessionMemory

# If your config.py exposes a PDF_PATH, great; otherwise set your own here:
try:
    from config import PDF_PATH
except Exception:
    PDF_PATH = Path("book.pdf")  # <-- adjust to your PDF if needed

# Build once
RETRIEVER = build_retriever(PDF_PATH)
TOOL = ChapterRetrieverTool(RETRIEVER)

LLM = build_llm()
# define_agents(tool) returns (router, summary, qa, quiz, feedback) in your codebase
ROUTER, SUMMARY_AGENT, QA_AGENT, QUIZ_AGENT, FEEDBACK_AGENT = define_agents(TOOL)

# simple session memory you already use in pdf_report.py
GLOBAL_MEM = SessionMemory()
