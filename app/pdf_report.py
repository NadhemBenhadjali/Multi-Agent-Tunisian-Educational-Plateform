from pathlib import Path
from typing import Any
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
from PIL import Image

from app.crew.config import ARABIC_FONT_NAME, IMG_DIR, MAX_IMG_W, MAX_IMG_H, MD_IMG
from app.helpers import rtl, strip_unsupported

class SessionMemory(dict):
    def log(self, k: str, v: Any):
        self[k] = v
        print(f"📝  خزّنا {k}.")

def render_pdf(mem: SessionMemory, outfile: Path) -> Path:
    """
    Renders a PDF report and now *also* draws any pictures that appear
    in mem['chapter_summary'] with the syntax ![alt](file-name.jpeg).
    """
    c = pdf_canvas.Canvas(str(outfile), pagesize=A4)
    w, h = A4
    margin_top, margin_bottom = 40, 40
    leading = 18
    y = h - margin_top                                         # cursor

    # ─ helpers ──────────────────────────────────────────────────────
    def new_page():
        nonlocal y
        c.showPage()
        y = h - margin_top

    def wrap_line(line, width=80):
        words, buf, out = line.split(), [], []
        for w_ in words:
            if sum(len(x) for x in buf) + len(w_) + len(buf) > width:
                out.append(" ".join(buf))
                buf = [w_]
            else:
                buf.append(w_)
        if buf:
            out.append(" ".join(buf))
        return out

    def draw_text(line: str, font=ARABIC_FONT_NAME, fsize=13):
        nonlocal y
        if y - leading < margin_bottom:
            new_page()
        c.setFont(font, fsize)
        c.drawRightString(w - margin_bottom, y, rtl(strip_unsupported(line)))
        y -= leading

    def draw_image(img_path: str, alt: str):
        nonlocal y
        full_path = f"{IMG_DIR}/{img_path}".replace(" ", "")
        try:
            im = Image.open(full_path)
        except FileNotFoundError:
            # fallback: just write the alt text as normal line
            draw_text(f"[صورة غير موجودة] {alt}")
            return

        # keep aspect ratio under MAX_IMG_W×MAX_IMG_H
        iw, ih = im.size
        scale = min(MAX_IMG_W / iw, MAX_IMG_H / ih, 1.0)
        dw, dh = iw * scale, ih * scale

        if y - dh - leading < margin_bottom:
            new_page()

        # draw under right margin (align on the right like text)
        c.drawInlineImage(full_path,
                          w - margin_bottom - dw,   # x
                          y - dh,                   # y (bottom)
                          width=dw,
                          height=dh)
        y -= dh + leading // 2
        # caption (alt) under the image
        draw_text(alt, font=ARABIC_FONT_NAME, fsize=11)

    # ─ generic block: can mix text + images ─────────────────────────
    def draw_rich_block(title: str, raw_md: str):
        nonlocal y
        draw_text(title, fsize=15)
        c.setStrokeColorRGB(0.6, 0.6, 0.6)
        c.line(margin_bottom, y + 6, w - margin_bottom, y + 6)
        y -= leading // 2

        for paragraph in raw_md.splitlines():
            paragraph = paragraph.strip()
            if not paragraph:
                y -= leading // 2
                continue

            # Does the whole line consist of a *single* markdown image?
            m = MD_IMG.fullmatch(paragraph)
            if m:
                alt, path = m.group(1).strip(), m.group(2).strip()
                draw_image(path, alt)
                continue

            # Otherwise treat as text (may *contain* inline images)
            # split by any image occurrences
            idx = 0
            for m in MD_IMG.finditer(paragraph):
                pre = paragraph[idx:m.start()].rstrip()
                if pre:
                    for l in wrap_line(pre):
                        draw_text(l)
                alt, path = m.group(1).strip(), m.group(2).strip()
                draw_image(path, alt)
                idx = m.end()
            tail = paragraph[idx:].rstrip()
            if tail:
                for l in wrap_line(tail):
                    draw_text(l)

        y -= leading // 2

    # ────────────────── COVER ──────────────────
    draw_text("📗 تقرير التعلّم", fsize=20)
    y -= leading

    # ────────────────── SUMMARY ─────────────────
    if "chapter_summary" in mem:
        draw_rich_block("ملخّص الدرس:", mem["chapter_summary"])

    # ────────────────── Q&A ─────────────────────
    if "qa_history" in mem:
        lines = []
        for q, a in mem["qa_history"]:
            lines.append(f"❓ {q}\n📥 {a}\n")
        draw_rich_block("الأسئلة و الأجوبة:", "\n".join(lines))

    # ────────────────── QUIZ DETAILS ───────────
    if "quiz_log" in mem:
        q_lines = []
        for idx, qd in enumerate(mem["quiz_log"], 1):
            mark = "✔" if qd["is_correct"] else "✘"
            q_lines.append(f"{idx}) {qd['q']}")
            if qd["type"] == "mc":
                q_lines.append("   " + "، ".join(qd["options"]))
            q_lines.append(f"   إجابتك: {qd['child']}   الصحيح: {qd['correct']}   {mark}\n")
        draw_rich_block("تفاصيل الاختبار:", "\n".join(q_lines))

    # ────────────────── SCORE & NOTE ────────────
    score_txt = ""
    if "quiz_results" in mem:
        r = mem["quiz_results"]
        pct = 100 * r["correct"] / (r["correct"] + r["incorrect"])
        score_txt = f"✅ {r['correct']}   ❌ {r['incorrect']}   النتيجة: {pct:.0f}%\n"
    if "feedback_note" in mem:
        score_txt += mem["feedback_note"]
    if score_txt:
        draw_rich_block("التقييم النهائي:", score_txt)

    c.save()
    return outfile
