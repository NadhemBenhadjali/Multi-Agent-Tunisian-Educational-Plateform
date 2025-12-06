"""
End-to-end pipeline for:
1. Writing a nested KG dict to Neo4j
2. Adding (:Image) nodes from a CSV
3. Computing & storing Arabic SBERT embeddings for every Lesson

All tasks run sequentially from the main block—no command-line
arguments required.
"""
# !pip install neo4j langchain-huggingface
from __future__ import annotations
from typing import Dict, Tuple

from pathlib import Path
import re
import pandas as pd
from neo4j import GraphDatabase, Driver
from langchain_huggingface import HuggingFaceEmbeddings

# ──────────────── Configuration ────────────────
NEO4J_URI:      str = "neo4j+s://2192f3cc.databases.neo4j.io"
NEO4J_USER:     str = "neo4j"
NEO4J_PASSWORD: str = "hLTWxx2CxBqDkp66-XR0n-GJyz48V8f3unBdd6rXQ8A"
CSV_PATH:       Path = Path("config_files") / "captions_ar (1).csv"
MODEL_NAME:     str = "Omartificial-Intelligence-Space/GATE-AraBert-v1"
CLEAR_DB_FIRST: bool = True          
# ────────────────────────────────────────────────


KG: Dict[str, Dict[str, Dict[str, Tuple[int, int]]]] = {
    "أحياء": {
        "الحواس": {
            "الحواس وأعضاء الحس":                (6,  11),
            "وظائف الجلد ووقايته":              (9,  12),
            "التأثيرات السلبية على حاستي السمع والإبصار": (12, 16),
            "حماية السمع والإبصار من المؤثرات المزعجة":    (16, 19),
            "تأثير مرض الزكام على الجسم":         (19, 22),
        },
        "التنقل": {
            "أنماط التنقل عند الحيوان":      (26, 30),
            "تكيف العضو مع نمط التنقل":       (30, 34),
        },
        "مصادر الأغذية": {
            "مسار الأغذية وتحولها داخل الأنيوب لحيوان عاشب": (36, 41),
            "أنواع الأسنان ووظائفها":                      (41, 44),
            "وقاية الأسنان":                                 (44, 48),
        },
        "التكاثر": {
            "التكاثر دون بذور": (56, 60),
        },
        "التنفس": {
            "أعضاء التنفس لدى بعض الحيوانات": (62, 66),
            "الرئتان عند الخروف":              (66, 69),
            "الغلاصم عند السمكة":              (69, 74),
        },
    },
    "فيزياء": {
        "الزمن": {
            "الساعة":  (78, 78),
            "الدقيقة": (81, 83),
            "الثانية": (84, 87),
        },
        "المادة": {
            "تعرف الهواء":          (92, 95),
            "إثبات وجود الهواء":    (95, 99),
            "خصائص الهواء":         (99, 102),
            "تلوث الهواء (1)":       (102, 105),
            "تلوث الهواء (2)":       (105, 108),
            "قيس كتل بواسطة الميزان": (108, 112),
        },
        "الطاقة": {
            "قوة الهواء تحدث عملا":              (118, 118),
            "الطاقة الحرارية وبعض مصادرها":        (118, 123),
            "المحافظة على دفء/برودة (العزل الحراري)": (123, 132),
            "تأثير الطاقة الحرارية تمددًا وتقلصًا":   (132, 137),
        },
    },
}
# ──────────────── Neo4j helpers ────────────────
def get_driver() -> Driver:
    """Return an authenticated Neo4j driver."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


class KGWriter:
    """
    Handles the Branch → Topic → Lesson hierarchy.
    (Branch)-[:HAS_TOPIC]->(Topic)-[:HAS_LESSON]->(Lesson)
    """

    def __init__(self, driver: Driver) -> None:
        self.driver = driver

    def clear_database(self) -> None:
        with self.driver.session() as sess:
            sess.run("MATCH (n) DETACH DELETE n")
        print("✅ cleared database")

    def write(self, kg: dict) -> None:
        with self.driver.session() as sess:
            for branch, topics in kg.items():
                sess.run("MERGE (:Branch {name:$name})", name=branch)

                for topic, lessons in topics.items():
                    sess.run(
                        """
                        MERGE (t:Topic {name:$topic})
                        WITH t
                        MATCH (b:Branch {name:$branch})
                        MERGE (b)-[:HAS_TOPIC]->(t)
                        """,
                        topic=topic,
                        branch=branch,
                    )

                    for title, (start, end) in lessons.items():
                        sess.run(
                            """
                            MERGE (l:Lesson {title:$title})
                            SET l.start_page=$s, l.end_page=$e
                            WITH l
                            MATCH (t:Topic {name:$topic})
                            MERGE (t)-[:HAS_LESSON]->(l)
                            """,
                            title=title,
                            s=start,
                            e=end,
                            topic=topic,
                        )
        print("✅ KG written")


# ──────────────── Image loader ────────────────
def _page_from_filename(fname: str) -> int | None:
    m = re.search(r"page_(\d+)", fname)
    return int(m.group(1)) if m else None


def add_images_from_csv(driver: Driver, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    with driver.session() as sess:
        for _, row in df.iterrows():
            fname   = str(row["image"])
            caption = str(row["caption_ar"])
            page    = _page_from_filename(fname)

            if page is None:
                print(f"⚠️  skip (no page): {fname}")
                continue

            sess.run(
                """
                MERGE (img:Image {name:$fname})
                SET img.caption=$caption, img.page=$page
                WITH img
                MATCH (l:Lesson)
                WHERE l.start_page <= $page AND $page <= l.end_page
                MERGE (l)-[:HAS_IMAGE]->(img)
                """,
                fname=fname,
                caption=caption,
                page=page,
            )
            print(f"✓ linked {fname} → {page}")

    print("✅ images processed")


# ──────────────── Embeddings ────────────────
def embed_lessons(driver: Driver, model_name: str) -> None:
    emb = HuggingFaceEmbeddings(model_name=model_name)

    with driver.session() as sess:
        for rec in sess.run("MATCH (l:Lesson) RETURN id(l) AS id, l.title AS title"):
            vec = emb.embed_query(rec["title"])
            sess.run(
                "MATCH (l) WHERE id(l)=$id "
                "SET l.vector_embedding=$vec",
                id=rec["id"],
                vec=vec,
            )
    print("✅ embeddings stored")


if __name__ == "__main__":
    driver = get_driver()
    try:
        kg_writer = KGWriter(driver)

        if CLEAR_DB_FIRST:
            kg_writer.clear_database()

        kg_writer.write(KG)
        add_images_from_csv(driver, CSV_PATH)
        embed_lessons(driver, MODEL_NAME)

        print("🎉 pipeline finished")
    finally:
        driver.close()
