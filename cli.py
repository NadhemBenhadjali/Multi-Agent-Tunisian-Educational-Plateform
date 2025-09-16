from pathlib import Path
import re
from IPython.display import Image as IPImage, display
import json
from crewai import Agent, Crew, Task, LLM
from langchain_huggingface import HuggingFaceEmbeddings
from handlers import _clean_json_block
from retrieval import build_retriever, ChapterRetrieverTool
from agents import define_agents
from images import fetch_lesson_images
from pdf_report import SessionMemory, render_pdf
from kg import Neo4jKG, _ask_user_for_topic
from utils_text import parse_quiz_json, _clean_user_question, cosine_similarity

def run_cli(pdf_path: Path, neo_kg: Neo4jKG,
            img_dir: Path = Path("config_files/book_images")) -> None:
    """
    pdf_path : Path to the scanned school-book PDF
    neo_kg   : Active Neo4jKG instance
    img_dir  : Folder that contains `page_XX_img_YY.jpeg` files
    """
    # 1) build Retriever tool + agents
    retriever = build_retriever(pdf_path)
    tool      = ChapterRetrieverTool(retriever)
    mem       = SessionMemory()
    router, summary, qa_agent, quiz_agent, feedback = define_agents(tool)

    # 2) little helper to pretty-print markdown with inline pictures ----------
    def render_with_images(markdown: str) -> None:
        for line in markdown.splitlines():
            m = re.search(r'!\[(.*?)\]\((.*?)\)', line)
            if m:
                # text before the image (usually the bullet):
                prefix   = line[:m.start()].rstrip()
                caption  = m.group(1).strip()
                file_nm  = m.group(2).strip()
                pic_path = img_dir / file_nm
                # show text
                print(prefix + (" " if prefix else "• ") + caption)
                # show image if it exists
                if pic_path.exists():
                    display(IPImage(filename=str(pic_path)))
                else:
                    print("⚠️  الصورة غير موجودة:", file_nm)
            else:
                print(line)

    # 3) main interactive loop ------------------------------------------------
    print(" مرحبا! اكتب 'ملخص …', 'سؤال: …', 'اختبرني', أو 'انهينا'.")
    while True:
        user_in = input("👦 » ").strip()
        if not user_in:
            continue

        # Decide which branch the child wants
        decision_task = Task(
            description=f"👂 إفهم طلب الطفل: «{user_in}». أرجع كلمة واحدة: summary | qa | quiz | end",
            expected_output="summary | qa | quiz | end",
            agent=router,
        )
        decision = (
            Crew(agents=[router], tasks=[decision_task], verbose=False)
            .kickoff()
            .raw.strip().lower()
        )

        # ──────── SUMMARY branch ────────────────────────────────────────────
        if decision == "summary":
            m = re.match(r"ملخص\s+(?:محور\s+)?(?P<topic>[\u0600-\u06FF ]+)", user_in)
            topic: str | None = m.group("topic").strip() if m else None
            if not topic:
                print("⚠️  لازم تذكر اسم المحور بعد كلمة «ملخص».")
                continue

            branch        = neo_kg.find_branch_for_topic(topic)
            lessons_info  = neo_kg.get_lessons_for_topic(topic)
            if not branch or not lessons_info:
                print(f" ما لقيتش المحور «{topic}» في الـ KG.")
                continue

            raw_chunks    : list[str] = []
            images_blocks : list[str] = []
            for lesson in lessons_info:
                raw_chunks.extend(tool.run(lesson["title"]))
                pics = fetch_lesson_images(neo_kg, lesson["title"])
                if pics:
                    md = "\n".join(
                        f"* [{p['caption']}]({p['name']})" for p in pics
                    )
                    images_blocks.append(f"درس «{lesson['title']}» – التصاور:\n{md}\n")

            ctx_text       = "\n".join(raw_chunks[:20])
            images_section = "\n".join(images_blocks) or "ما ثـمّـة حتى تصاور."
            sub_lessons_md = "\n".join(f"• {ld['title']}" for ld in lessons_info)

            # ── upgraded, more-pedagogical summary prompt ───────────────────────────
            sum_prompt = f"""
            إنتي معلّم/ة تونسي/ة؛ هدفك تبسّط محور “{topic}” من فرع “{branch}” لتلميذ في
            السنة الرابعة ابتـدائي. ركّز على الفهم، ربط الأفكار بحياتو اليومية،
            وتنويع الأمثلة.

            المعطيات قدامك:
            ┌─ الدروس الفرعيّة:
            {sub_lessons_md}

            ┌─ مقتطفات من الكتاب (تستعملها كان تحب تقتبس جملة ولا توضيح):
            {ctx_text}

            ┌─ مجموعة تصاور مرتبطة (اختياري تستعمل بعضها):
            {images_section}

            طريقة العمل المطلوبة:
            1) إفتتاحيّة صغيرة بالدارجة (سطرين ـ ٣ سطور) تعرّف فيها بالمحور
               ولماذا يهمّ التلميذ في حياتو.
            2) بعد الإفتتاحيّة، امشِ درس درس:
                  • إبدأ السطر بالرمز «•».
                  • اشرح الفكرة الرئيسية بعبارة مبسّطة.
                  • أعط مثال واقعي من حياة الطفل (الدار، الحومة، الطبيعة…).
                  • أذكر المنفعة أو الهدف التربوي (علاش يلزم يعرفها).
                  • إن لزمَ الأمر وتوجد صورة توضّح المقصود، أدرجها هكذا:
                    ![وصف مختصر](file-name.jpeg)
                    ↳ استعمل أقصى حد ٣ صور في كامل الملخّص، وما تحطّش صورة
                      كان الشرح وحدو كافي.
            3) أختم بسطر يُلخّص «رسالة/عبرة» المحور، وتحفيز صغير باش يراجع الدرس.

            تنبيهات أسلوبية:
            - أكتب باللهجة التونسية الخفيفة، جُمَل قصيرة، مفردات مألوفة.
            - استخدم أفعال أمر إيجابية: «جرّب»، «ركّز»، «لاحظ».
            - ما تذكرش أرقام الصفحات، ولا أسماء الملفات في النص (إلا في
              الصيغة ![alt](path) وقت تستعمل صورة).
            """

            sum_task = Task(description=sum_prompt,
                            expected_output="markdown with bullets & images",
                            agent=summary)

            md_out = Crew(agents=[summary], tasks=[sum_task],
                          verbose=False).kickoff().raw
            mem.log("chapter_summary", md_out)
            render_with_images(md_out)

        # ── “سؤال: …” (QA) branch ──────────────────────────────────────────────────
        elif decision == "qa":
            question = _clean_user_question(user_in)

            # 1) احسب embedding للسؤال
            hf_emb = HuggingFaceEmbeddings(model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1")
            q_embedding = hf_emb.embed_query(question)

            # 2) تجيب كل دروس المحفظة مع الـ embeddings متاعهم
            all_lessons = neo_kg.fetch_all_lesson_embeddings()

            # 3) احسب أعلى تشابه
            best_score = -1.0
            inferred_topic = None
            inferred_lesson = None
            for entry in all_lessons:
                lesson_embed = entry["embedding"]
                score = cosine_similarity(q_embedding, lesson_embed)
                if score > best_score:
                    best_score = score
                    inferred_topic = entry["topic"]
                    inferred_lesson = entry["lesson"]
            print("inferedtopic",inferred_topic)
            print("inferred_lesson",inferred_lesson)
            # 4) إذا التشابه أقل من threshold (مثلاً 0.25)، نطلب من الطفل يحدد المحور يدويًا
            if inferred_topic is None or best_score < 0.25:
                print("ما فهمتش المحور . على أي محور تسأل؟")
                inferred_topic = _ask_user_for_topic(neo_kg)
                inferred_lesson=inferred_topic
                if not inferred_topic:
                    print("حسنًا، نرجع للقائمة الرئيسية.")
                    continue
                # يمكنك تخطي inferred_lesson أو جعله None → سنعتمد على inferred_topic فقط
            print("inferred_topic2",inferred_topic)
            lessons_info = neo_kg.get_lessons_for_topic(inferred_topic)
            raw_chunks = []
            for lesson in lessons_info:
                raw_chunks.extend(tool.run(lesson["title"]))

            ctx_text = "\n".join(raw_chunks[:20])
            branch = neo_kg.find_branch_for_topic(inferred_topic)

            # 6) بناء Prompt بيداغوجي باللهجة التونسية
            sub_lessons_list = "\n".join(
                f"• {ld['title']} (pages {ld['start_page']}–{ld['end_page']})"
                for ld in lessons_info
            )
            qa_prompt = (
                f"أنت معلّم صبور ولطيف. وصلتك هذه السؤال من الطفل:\n"
                f"«{question}»\n\n"
                f"درس “{inferred_lesson or '‹اخترت المحور دون تحديد درس›'}” تحت محور “{inferred_topic}” هو الأنسب.\n\n"
                f"هذه قائمة جميع الدروس في المحور:\n{sub_lessons_list}\n\n"
                "قدّم شرحًا تفصيليًا بعبارات بسيطة ومفهومة:\n"
                "- عرّف المصطلح أو الطريقة المطلوبة.\n"
                "- أضف مثالًا صغيرًا من الحياة اليومية.\n"
                "- إذا في جزء صعيب، فسّره بخطوات بسيطة كأنك تشرح لتلميذ في الصف الرابع.\n\n"
                "ما تذكرش أرقام الصفحات. أجب بلهجة دارجة تونسية."
            )
            qa_task = Task(
                description=qa_prompt,
                expected_output="جواب …",
                agent=qa_agent,
            )

            answer = Crew(agents=[qa_agent], tasks=[qa_task], verbose=False).kickoff().raw
            mem.setdefault("qa_history", []).append((question, answer))
            print(answer)

        # ── “اختبرني” (Quiz) branch ─────────────────────────────────────────────────
        elif decision == "quiz":
            print("🔢 على أي محور تحب أن تختبر نفسك؟")
            chosen_topic = _ask_user_for_topic(neo_kg)
            if not chosen_topic:
                print("حسنًا، عدنا إلى القائمة الرئيسية.")
                continue

            # 2) Fetch that topic’s lessons
            lessons_info = neo_kg.get_lessons_for_topic(chosen_topic)
            if not lessons_info:
                print(f"⚠️ لا توجد دروس تحت الموضوع «{chosen_topic}».")
                continue

            branch = neo_kg.find_branch_for_topic(chosen_topic)
            # 3) Gather raw text for each lesson
            raw_chunks = []
            for lesson in lessons_info:
                raw_chunks.extend(tool.run(lesson["title"]))

            ctx_text = "\n".join(raw_chunks[:30])  # a bit more to cover full chapter

            # 4) Build a Quiz prompt: “Create 10 questions (5 MC, 5 T/F) covering all points”
            sub_lessons_list = "\n".join(f"• {ld['title']} (pages {ld['start_page']}–{ld['end_page']})"
                                         for ld in lessons_info)
            quiz_prompt = (
                f"You are a quiz‐maker for primary school. Create a JSON containing **10 questions** "
                f"about topic “{chosen_topic}” under branch “{branch}.”\n"
                "– 6 multiple‐choice questions (type='mc'), each with 4 options and exactly one correct answer.\n"
                "– 4 true/false questions (type='tf'), each with 'صح' or 'خطأ' as the correct answer.\n\n"
                "Make sure the questions cover **all sub‐lessons** listed below, and reference page ranges if needed:\n"
                f"{sub_lessons_list}\n\n"
                f"Here are some raw text snippets from the chapter:\n{ctx_text}\n\n"
                "Return exactly a JSON with structure:\n"
                "{ 'questions': [ {'type':'mc','q':'…','options':['…','…','…','…'],'a':'…'}, … ] }\n"
            )
            quiz_task = Task(
                description=quiz_prompt,
                expected_output="json",
                agent=quiz_agent,
            )

            raw_json = Crew(agents=[quiz_agent], tasks=[quiz_task], verbose=False).kickoff().raw
            quiz_data = parse_quiz_json(raw_json)
            if not quiz_data or "questions" not in quiz_data:
                print("❗ خطأ في JSON المولَّد.")
                continue

            # 5) Run through the quiz with the child
            correct = incorrect = 0
            mem["quiz_log"] = []
            for idx, qobj in enumerate(quiz_data["questions"], 1):
                print(f"\n{idx} - {qobj['q']}")
                if qobj["type"] == "mc":
                    print("   الخيارات:", ", ".join(qobj["options"]))
                elif qobj["type"] == "tf":
                    print("   الخيارات: صحيح / خطأ")

                # Get child’s valid answer
                allowed = set(opt.lower() for opt in qobj.get("options", []))
                if qobj["type"] == "tf":
                    allowed |= {"صح", "خطأ", "true", "false", "t", "f"}
                while True:
                    ans = input("   ➜ إجابتك: ").strip().lower()
                    if ans in allowed:
                        break
                    print(" اختَر من الخيارات المعروضة فقط.")

                # Normalize true/false
                user_ans = ans
                if qobj["type"] == "tf":
                    user_ans = "صح" if ans in {"t", "صح", "true"} else "خطأ"
                correct_ans = qobj["a"].strip().lower()
                if qobj["type"] == "tf":
                    correct_ans = "صح" if qobj["a"].strip().lower() in {"t", "true", "صح"} else "خطأ"

                is_ok = user_ans == correct_ans
                correct += is_ok
                incorrect += not is_ok
                mem["quiz_log"].append({
                    "q": qobj["q"],
                    "type": qobj["type"],
                    "options": qobj.get("options"),
                    "child": user_ans,
                    "correct": correct_ans,
                    "is_correct": is_ok,
                })
                print("✔" if is_ok else f"✘ الإجابة الصحيحة: {correct_ans}")

            # 6) Record final results + feedback
            mem["quiz_results"] = {"correct": correct, "incorrect": incorrect}
            ratio = correct / (correct + incorrect)
            mem["session_passed"] = ratio >= 8                
            mem["quiz_rating"] = ratio
            if ratio == 1.0:
                mem["feedback_note"] = "ممتاز! 🌟 حافظ على هذا المستوى!"
            elif ratio >= 0.7:
                mem["feedback_note"] = "عمل طيب! راجع الأخطاء وحاول مرة أخرى."
            else:
                mem["feedback_note"] = "لا تقلق، الأهم هو المحاولة. سنراجع معًا. 💪"

            print(f"\n🔢 نتيجتك: ✅ {correct}  ❌ {incorrect}")
            print(mem["feedback_note"])

        # ── “انهينا” (end) branch ───────────────────────────────────────────────────
        elif decision == "end":
            # 1) Build a prompt for the feedback agent, using all logged items in mem
            fb_parts = []

            # If there was a summary, include it
            if "chapter_summary" in mem:
                fb_parts.append(
                    "ملخّص الدرس:\n" + mem["chapter_summary"] + "\n"
                )

            # If there is QA history, include those exchanges
            if "qa_history" in mem:
                qa_lines = []
                for q, a in mem["qa_history"]:
                    qa_lines.append(f"❓ {q}\n📥 {a}\n")
                fb_parts.append("الأسئلة و الأجوبة:\n" + "\n".join(qa_lines) + "\n")

            # If the user took a quiz, include the quiz results
            if "quiz_results" in mem and "quiz_log" in mem:
                qr = mem["quiz_results"]
                ratio = (qr["correct"] / (qr["correct"] + qr["incorrect"])) * 100
                quiz_summary = (
                    f"نتيجة الاختبار: ✅ {qr['correct']}  ❌ {qr['incorrect']}  (النسبة: {ratio:.0f}%)\n"
                )
                fb_parts.append("تفاصيل الاختبار:\n" + quiz_summary + "\n")
            if "quiz_rating" in mem:
                fb_parts.append(
                    "ملخص الاختبار:\n"
                    f"- عدد الأسئلة: {mem['quiz_results']['correct'] + mem['quiz_results']['incorrect']}\n"
                    f"- صحيحة: {mem['quiz_results']['correct']}\n"
                    f"- خاطئة: {mem['quiz_results']['incorrect']}\n"
                    f"- تقييم على 10: {mem['quiz_rating']}\n"
                    f"- النتيجة النهائية: {'ناجح' if mem['session_passed'] else 'لم يمر بعد'}\n"
                )
            # # 2) Create the feedback prompt
            # fb_prompt = (
            #     "أنت أخصّائي متابعة تعلّم للأطفال. "
            #     "بناءً على هذه المعلومات من جلستهم:\n\n"
            #     + "\n---\n".join(fb_parts)
            #     + "\n\n"
            #     "مهمّتك مزدوجة:\n"
            #     "1. اكتب رسالة تشجيعية قصيرة باللهجة التونسية، تلخّص نجاحاتهم وتشجّعهم على الاستمرار في الدراسة.\n"
            #     "2. أنشئ أيضًا كائناً JSON يحتوي على ملخّص منظم للجلسة وفق هذا الشكل:\n\n"
            #     "{\n"
            #     # '  "session_id": "SESSION_ID_HERE",\n'
            #     # '  "date": "تاريخ الجلسة",\n'
            #     '  "branch": "المجال أو المادة",\n'
            #     '  "topic": "الموضوع العام",\n'
            #     '  "lesson": "عنوان الدرس",\n'
            #     '  "summary": "محتوى الجلسة باختصار",\n'
            #     '  "steps": [ "خطوة 1", "خطوة 2", "..." ],\n'
            #     '  "feedback": "ملاحظات عن أداء الطفل",\n'
            #     '  "quiz_rating": تقييم رقمي من 1 إلى 10,\n'
            #     '  "session_passed": true أو false\n'
            #     '  "feedback": "ملاحظات عن أداء الطفل",\n'
            #     '  "quiz_rating": 0–10,\n'
            #     '  "session_passed": true/false,\n'
            #     '  "encouragement": "رسالة باللهجة التونسية تشجع الطفل"\n'
            #     "}\n"
            #     "### تقرير الجلسة (JSON)\n"
            # )
            fb_prompt = (
                "أنت أخصّائي متابعة تعلّم للأطفال. "
                "استعمل المعلومات الآتية عن الجلسة، ثم أرجِع **كائناً JSON واحداً صالحاً** فقط:\n\n"
                + "\n---\n".join(fb_parts)
                + "\n\n"
                "هيكل JSON المطلوب:\n"
                "{\n"
                '  "branch":        "المجال أو المادة",\n'
                '  "topic":         "الموضوع العام",\n'
                '  "lesson":        "عنوان الدرس (إن وجد)",\n'
                '  "summary":       "ملخص قصير للجلسة",\n'
                '  "steps":         ["خطوة 1", "خطوة 2", "..."],\n'
                f'  "quiz_rating":   {mem.get("quiz_rating", None)},\n'
                f'  "session_passed": {str(mem.get("session_passed", False)).lower()},\n'
                '  "feedback":      " **رسالة موجهة للمعلّم/ة**: تحليل نقاط القوة، الصعوبات، '
                'ونصائح عملية لتحسين التعلّم",\n'                
                '  "encouragement": "رسالة باللهجة التونسية تشجّع الطفل"\n'
                "}\n\n"
                "انتبه:\n"
                "- في \"feedback\" اشرح  لماذا أخطأ التلميذ، وكيف يتجاوز الصعوبات.\n"
                "- لا تطبع أي شيء خارج قوسَي JSON."
            )


            feedback_task = Task(
                description=fb_prompt,
                expected_output="JSON ملخّص للجلسة",
                agent=feedback,
            )

            fb_note = Crew(
                agents=[feedback],
                tasks=[feedback_task],
                verbose=False
            ).kickoff().raw
            
            cleaned = _clean_json_block(fb_note)
            session_report = json.loads(cleaned)
            mem["session_report"] = session_report        
            mem["feedback_note"]  = session_report["encouragement"] 

            # 4) Now render the PDF (it will pick up mem['feedback_note'] automatically)
            pdf_file = render_pdf(mem, Path("session_report.pdf"))
            print("التقرير جاهز:", pdf_file)
            break

        else:
            print(" ممشيتش الأمور، جرّب كلمة أخرى.")

    # When loop ends, close the Neo4j connection
    neo_kg.close()
