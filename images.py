from typing import List
from kg import Neo4jKG

def fetch_lesson_images(kg: Neo4jKG, lesson_title: str) -> list[dict]:
    """
    Return every Image attached to a Lesson via
    (l:Lesson)-[:HAS_IMAGE]->(img:Image).
    Each row is a dict with keys: file, caption, page.
    """
    cypher = """
    MATCH (l:Lesson {title: $title})-[:HAS_IMAGE]->(img:Image)
    RETURN img.name    AS name,
           img.caption AS caption,
           img.page    AS page
    ORDER BY img.page
    """
    with kg.driver.session() as session:
        return session.run(cypher, title=lesson_title).data()
