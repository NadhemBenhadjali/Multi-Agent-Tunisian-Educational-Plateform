from __future__ import annotations
import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Make sure OCR env is good on Windows BEFORE importing config
try:
    import pytesseract, pathlib
    if os.name == "nt":
        os.environ.setdefault(
            "TESSDATA_PREFIX",
            str((pathlib.Path(__file__).resolve().parents[1] / "tessdata").resolve())
        )
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    pass

from config import URI, USER, PASSWORD
from kg import Neo4jKG
from pdf_report import render_pdf

from runtime import GLOBAL_MEM
from handlers import generate_summary_json, handle_qa, generate_quiz_json

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# Where to serve saved JSON/Reports from:
LESSONS_DIR = "lessons"
REPORTS_DIR = "reports"
os.makedirs(LESSONS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
app.mount("/lessons", StaticFiles(directory=LESSONS_DIR), name="lesson_files")
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

neo_kg = Neo4jKG(URI, USER, PASSWORD)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/summary")
async def summary_endpoint(req: Request):
    body = await req.json()
    mod = body.get("module", "").strip()
    if not mod:
        return JSONResponse({"error": "module is required"}, status_code=400)
    try:
        user_in = f"ملخص محور {mod}"
        result  = generate_summary_json(user_in, neo_kg)
        GLOBAL_MEM.log("chapter_summary", result["data"])
        return JSONResponse(result)
    except LookupError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": "internal failure", "details": str(e)}, status_code=500)

@app.post("/qa")
async def qa_endpoint(req: Request):
    body     = await req.json()
    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)
    try:
        # reuse the same embedding instance as your retriever (imported inside handler)
        from runtime import RETRIEVER
        answer = handle_qa(question, neo_kg, RETRIEVER.base_retriever.vectorstore._embedding_function)
        GLOBAL_MEM.log("qa_history", (question, answer))
        return JSONResponse(answer)
    except LookupError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": "internal failure", "details": str(e)}, status_code=500)

@app.post("/quiz")
async def quiz_endpoint(req: Request):
    body   = await req.json()
    module = body.get("module", "").strip()
    num_mc = int(body.get("num_mc", 6))
    num_tf = int(body.get("num_tf", 4))
    if not module:
        return JSONResponse({"error": "module is required"}, status_code=400)
    try:
        result = generate_quiz_json(module, neo_kg, num_mc=num_mc, num_tf=num_tf)
        GLOBAL_MEM["quiz_log"] = result["data"]["questions"]
        GLOBAL_MEM["quiz_results"] = {"correct": 0, "incorrect": len(result["data"]["questions"])}
        return JSONResponse(result)
    except LookupError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": "internal failure", "details": str(e)}, status_code=500)

@app.post("/finish")
async def finish():
    # assemble a light feedback prompt and render the PDF report
    parts = []
    if "chapter_summary" in GLOBAL_MEM:
        parts.append("ملخّص الدرس:\n" + str(GLOBAL_MEM["chapter_summary"]))
    if "qa_history" in GLOBAL_MEM:
        qa_lines = [f"❓ {q}\n📥 {a}" for q, a in GLOBAL_MEM["qa_history"]]
        parts.append("الأسئلة و الأجوبة:\n" + "\n".join(qa_lines))
    if "quiz_log" in GLOBAL_MEM:
        quiz_lines = [f"{i+1}) {q.get('q')} – الصحيح: {q.get('a')}" for i, q in enumerate(GLOBAL_MEM["quiz_log"])]
        parts.append("تفاصيل الاختبار:\n" + "\n".join(quiz_lines))

    fb_prompt = (
        "أنت أخصّائي متابعة تعلم.\n"
        + "\n---\n".join(parts)
        + "\nاكتب رسالة تشجيعية قصيرة باللهجة التونسية."
    )
    from crewai import Crew, Task
    from runtime import FEEDBACK_AGENT
    fb_task = Task(description=fb_prompt, expected_output="رسالة تشجيعية", agent=FEEDBACK_AGENT)
    fb_note = Crew(agents=[FEEDBACK_AGENT], tasks=[fb_task], verbose=False).kickoff().raw
    GLOBAL_MEM["feedback_note"] = fb_note

    # render PDF into ./reports
    from pathlib import Path
    pdf_path = Path(REPORTS_DIR) / "session_report.pdf"
    render_pdf(GLOBAL_MEM, pdf_path)

    return JSONResponse({"pdf_url": "/reports/session_report.pdf"})
