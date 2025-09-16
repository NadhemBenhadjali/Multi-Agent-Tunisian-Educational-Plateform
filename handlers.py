from __future__ import annotations
import json, re
from typing import Tuple, Any, List
from crewai import Crew, Task

from images import fetch_lesson_images
from retrieval import ChapterRetrieverTool  # for type hints only
from pdf_report import render_pdf
from kg import Neo4jKG

from runtime import SUMMARY_AGENT, QA_AGENT, QUIZ_AGENT, FEEDBACK_AGENT, TOOL, GLOBAL_MEM

# ——— small helpers (kept in-file to avoid touching your utils) ———
def _clean_user_question(raw: str) -> str:
    l = raw.strip().lower()
    return raw.split(':',1)[1].strip() if l.startswith(('سؤال:','qa:')) else raw.strip()

def _clean_json_block(text: str) -> str:
    import re
    cleaned = re.sub(r"```[a-zA-Z]*\n?", "", text).strip()
    return cleaned.strip("`").strip()

def parse_quiz_json(raw_text: str):
    import json, re, ast
    cleaned = _clean_json_block(raw_text).replace("'", '"')
    try:
        return json.loads(cleaned)
    except Exception:
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            return None

# ——— shared retrieval of context & images ———
def retrieve_context(topic: str, kg: Neo4jKG) -> Tuple[str, str]:
    lessons = kg.get_lessons_for_topic(topic)
    text_chunks: List[str] = []
    images_blocks: List[str] = []
    for ld in lessons:
        text_chunks.extend(TOOL.run(ld["title"]))
        pics = fetch_lesson_images(kg, ld["title"])
        if pics:
            md = "\n".join(f"* [{p['caption']}]({p['name']})" for p in pics)
            images_blocks.append(f"درس «{ld['title']}» – التصاور:\n{md}\n")
    return "\n".join(text_chunks[:30]), ("\n".join(images_blocks) or "ما ثـمّـة حتى تصاور.")

# ——— SUMMARY ———
def generate_summary_json(user_in: str, kg: Neo4jKG) -> dict:
    m = re.match(r"ملخص\s+(?:محور\s+)?(?P<topic>[\u0600-\u06FF ]+)", user_in)
    if not m:
        raise ValueError("⚠️ لازم تذكر اسم المحور بعد كلمة «ملخص».")
    topic = m.group("topic").strip()

    branch = kg.find_branch_for_topic(topic)
    lessons_info = kg.get_lessons_for_topic(topic)
    if not branch or not lessons_info:
        raise LookupError(f"⚠️ ما لقيتش المحور «{topic}» في الـ KG.")

    ctx_text, images_section = retrieve_context(topic, kg)
    sub_lessons_md = "\n".join(f"• {ld['title']}" for ld in lessons_info)

    prompt = f"""
إنتي معلّم/ة تونسي/ة؛ هدفك تبسّط محور “{topic}” من فرع “{branch}” لتلميذ في
السنة الرابعة ابتـدائي. ركّز على الفهم، ربط الأفكار بحياتو اليومية، وتنويع الأمثلة.

المعطيات قدامك:
┌─ الدروس الفرعيّة:
{sub_lessons_md}

┌─ مقتطفات من الكتاب (تستعملها كان تحب تقتبس جملة ولا توضيح):
{ctx_text}

┌─ مجموعة تصاور مرتبطة (اختياري تستعمل بعضها):
{images_section}

طريقة العمل المطلوبة:
1) إفتتاحيّة صغيرة بالدارجة (سطرين ـ ٣ سطور) تعرّف فيها بالمحور ولماذا يهمّ التلميذ.
2) بعد الإفتتاحيّة، امشِ درس درس:
   • اشرح الفكرة الرئيسية بعبارة مبسّطة.
   • أعط مثال واقعي من حياة الطفل.
   • إذا لزم الأمر وتوجد صورة، أدرجها هكذا: ![alt](file-name.jpeg) (حد أقصى ٣ صور).
3) أختم بسطر يُلخّص «رسالة/عبرة» المحور.

أخرج JSON فقط بالهيكل:
{{
  "title": "درس عن {topic}",
  "slides": [
    {{ "number": "1", "text": "..." }},
    {{ "number": "2", "text": "..." }}
  ]
}}
""".strip()

    task = Task(description=prompt, expected_output="json", agent=SUMMARY_AGENT)
    raw = Crew(agents=[SUMMARY_AGENT], tasks=[task], verbose=False).kickoff().raw
    cleaned = _clean_json_block(raw)
    start = cleaned.find("{"); end = cleaned.rfind("}")
    if start < 0 or end < 0:
        from fastapi import HTTPException
        raise HTTPException(502, "No JSON object found in LLM output")

    data = json.loads(cleaned[start:end+1])
    filename = f"{branch}_{topic}.json".replace(" ", "_")

    # where to write lesson JSON (match your existing pattern if you have one)
    from pathlib import Path
    from config import Path as _Path  # just to avoid shadowing
    out_dir = Path("lessons"); out_dir.mkdir(exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"path": f"/lessons/{filename}", "data": data}

# ——— QA ———
def handle_qa(question: str, kg: Neo4jKG, emb) -> str:
    q = _clean_user_question(question)
    # basic vector-based topic pick using your KG embeddings
    q_emb = emb.embed_query(q)
    all_lessons = kg.fetch_all_lesson_embeddings()

    def _cos(a, b):
        import math
        dot = sum(x*y for x, y in zip(a, b))
        n1 = math.sqrt(sum(x*x for x in a)); n2 = math.sqrt(sum(y*y for y in b))
        return dot/(n1*n2) if n1 and n2 else 0.0

    best_score, inferred_topic, inferred_lesson = -1.0, None, None
    for row in all_lessons:
        score = _cos(q_emb, row["embedding"])
        if score > best_score:
            best_score, inferred_topic, inferred_lesson = score, row["topic"], row["lesson"]

    if best_score >= 0.25 and inferred_topic:
        ctx_text, _ = retrieve_context(inferred_topic, kg)
        sub_md = "\n".join(f"• {ld['title']}" for ld in kg.get_lessons_for_topic(inferred_topic))
        prompt = (
            f"أنت معلّم صبور. السؤال: «{q}»\n"
            f"الدرس الأنسب: “{inferred_lesson}” تحت محور “{inferred_topic}”.\n"
            f"الدروس:\n{sub_md}\n\n"
            "اشرح ببساطة مع مثال من الحياة اليومية."
        )
    else:
        prompt = (
            f"أنت معلّم صبور. السؤال: «{q}»\n"
            "اشرح ببساطة مع مثال من الحياة اليومية."
        )

    task = Task(description=prompt, expected_output="text", agent=QA_AGENT)
    answer = Crew(agents=[QA_AGENT], tasks=[task], verbose=False).kickoff().raw
    return answer

# ——— QUIZ ———
def generate_quiz_json(module: str, kg: Neo4jKG, num_mc: int = 6, num_tf: int = 4) -> dict:
    branch = kg.find_branch_for_topic(module)
    lessons_info = kg.get_lessons_for_topic(module)
    if not branch or not lessons_info:
        raise LookupError(f"⚠️ ما لقيتش المحور «{module}» في الـ KG.")
    ctx_text, _ = retrieve_context(module, kg)
    sub_list = "\n".join(f"• {ld['title']} (pages {ld['start_page']}–{ld['end_page']})" for ld in lessons_info)

    prompt = (
        f"أنت صانع امتحانات لابتدائي. أعد JSON فيه {num_mc} MC و{num_tf} صح/خطأ "
        f"عن محور «{module}» (فرع «{branch}»). غطّ كل الدروس:\n{sub_list}\n\n"
        f"مقتطفات:\n{ctx_text}\n\n"
        "Return JSON: { 'questions': [ {'type':'mc',...}, {'type':'tf',...} ] }"
    )
    task = Task(description=prompt, expected_output="json", agent=QUIZ_AGENT)
    raw = Crew(agents=[QUIZ_AGENT], tasks=[task], verbose=False).kickoff().raw
    data = parse_quiz_json(raw)
    return {"module": module, "data": data}
