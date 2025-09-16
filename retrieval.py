from typing import Any, List, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

from ocr_pdf import load_arabic_pdf

# ─────────────────────────── Build Retriever ─────────────────────────
def build_retriever(pdf_path, embedding_model="Omartificial-Intelligence-Space/GATE-AraBert-v1", reranker_model="Omartificial-Intelligence-Space/ARA-Reranker-V1", k_fetch=8, k_rerank=3):
    docs = load_arabic_pdf(pdf_path)
    emb = HuggingFaceEmbeddings(model_name=embedding_model)
    chunks = SemanticChunker(emb).split_documents(docs)
    vect = Chroma.from_documents(chunks, emb)
    base_ret = vect.as_retriever(search_kwargs={"k": k_fetch})
    cross = HuggingFaceCrossEncoder(model_name=reranker_model)
    comp = CrossEncoderReranker(model=cross, top_n=k_rerank)
    return ContextualCompressionRetriever(base_compressor=comp, base_retriever=base_ret)

# ─────────────────────── ChapterRetriever Tool ──────────────────────
class ChapterRetrieverInput(BaseModel):
    query: str = Field(..., description="السؤال أو عنوان الدرس")

class ChapterRetrieverTool(BaseTool):
    name: str = "chapter_retriever"
    description: str = "يجيب مقاطع من الكتاب حسب السؤال أو عنوان الدرس."
    args_schema: Type[BaseModel] = ChapterRetrieverInput

    def __init__(self, retriever: ContextualCompressionRetriever):
        super().__init__()
        object.__setattr__(self, "_retriever", retriever)

    def _run(self, query: str, **kwargs: Any) -> List[str]:
        print(f"Retrieving pages for «{query}» …")
        return [d.page_content for d in self._retriever.invoke(query)]
