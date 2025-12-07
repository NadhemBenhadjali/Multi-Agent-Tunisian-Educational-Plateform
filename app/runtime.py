# -*- coding: utf-8 -*-
# config_files/api/runtime.py
from __future__ import annotations
from crewai_tools import QdrantVectorSearchTool
from app.crew.agents import build_llm, define_agents
from app.pdf_report import SessionMemory
import os
from app.helpers import embed

TOOL = QdrantVectorSearchTool(
    qdrant_url=os.getenv("QDRANT_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    collection_name="etudeai",
    limit=5,
    score_threshold=0.35,
    custom_embedding_fn=embed
)
LLM = build_llm()
SUMMARY_AGENT, QA_AGENT, QUIZ_AGENT, FEEDBACK_AGENT = define_agents(TOOL)
# simple session memory you already use in pdf_report.py
GLOBAL_MEM = SessionMemory()
