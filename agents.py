from crewai import Agent, LLM

def build_llm() -> LLM:
    return LLM(model="gemini/gemini-2.0-flash", temperature=0.5, max_tokens="4000")

def define_agents(tool) -> tuple[Agent, Agent, Agent, Agent, Agent]:
    llm = build_llm()

    router = Agent(
        role="Routeur",
        goal="تفهم طلب الطفل و توجّهو",
        backstory="معلّم يرتّب الخدمة.",
        llm=llm,
        verbose=True,
    )
    summary = Agent(
        role="ملخّص الدرس",
        goal="ملخّص ساهل بالدارجة",
        backstory="معلّمة توضّح الدروس.",
        llm=llm,
        tools=[tool],
        verbose=True,
    )
    qa = Agent(
        role="معلّم يجاوب",
        goal="يشرح بصبر ويتأكّد من الفهم ويستشهد بالصفحات",
        backstory="معلّم صبور.",
        llm=llm,
        tools=[tool],
        verbose=True,
    )
    quiz = Agent(
        role="صانع الامتحانات",
        goal="يعمل أسئلة بسيطة ويصحّحها بناءً على المحور المحدّد",
        backstory="يحب النجوم الذهبية.",
        llm=llm,
        tools=[tool],
        verbose=True,
    )
    feedback = Agent(
        role="معدّ التقرير",
        goal="يكتب تقرير PDF مشجّع",
        backstory="أخصّائي متابعة تعلم.",
        llm=llm,
        verbose=True,
    )
    return router, summary, qa, quiz, feedback
