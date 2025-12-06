from neo4j import GraphDatabase
from typing import List

class Neo4jKG:
    def __init__(self, uri: str, user: str, pwd: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        self.driver.close()

    def get_lessons_for_topic(self, topic_name: str) -> list[dict]:
        """
        Returns a list of dicts:
          [{ 'title': <string>, 'start_page': <int>, 'end_page': <int> }, …]
        """
        query = """
        MATCH (t:Topic {name: $topic_name})-[:HAS_LESSON]->(l:Lesson)
        RETURN l.title AS title, l.start_page AS start_page, l.end_page AS end_page
        ORDER BY l.title
        """
        with self.driver.session() as session:
            result = session.run(query, topic_name=topic_name)
            return [record.data() for record in result]

    def find_branch_for_topic(self, topic_name: str) -> str | None:
        """
        Returns the parent Branch name of a given topic, or None if not found.
        """
        query = """
        MATCH (b:Branch)-[:HAS_TOPIC]->(t:Topic {name: $topic_name})
        RETURN b.name AS branch_name
        """
        with self.driver.session() as session:
            rec = session.run(query, topic_name=topic_name).single()
            return rec["branch_name"] if rec else None

    def list_all_topics(self) -> list[str]:
        """
        Returns the list of all topic names currently in the KG.
        """
        query = "MATCH (t:Topic) RETURN t.name AS name ORDER BY t.name"
        with self.driver.session() as session:
            result = session.run(query)
            return [record["name"] for record in result]
    def fetch_all_lesson_embeddings(self) -> list[dict]:
        """
        Return a list of dicts, each containing:
          - 'topic': parent topic name
          - 'lesson': lesson title
          - 'embedding': the stored vector_embedding (list of floats)
        """
        cypher = """
        MATCH (t:Topic)-[:HAS_LESSON]->(l:Lesson)
        WHERE l.vector_embedding IS NOT NULL
        RETURN t.name AS topic, l.title AS lesson, l.vector_embedding AS embedding
        """
        with self.driver.session() as session:
            records = session.run(cypher)
            return [record.data() for record in records]
    def fetch_lesson_images(self,lesson_title: str) -> list[dict]:
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
        with self.driver.session() as session:
            return session.run(cypher, title=lesson_title).data()
    def extract_images(self, topic: str) -> str:
        """
        Builds a markdown block listing images grouped by lesson for the given topic.
        Returns a single markdown string.
        """
        lessons = self.get_lessons_for_topic(topic)
        images_blocks: List[str] = []
        for ld in lessons:
            pics = self.fetch_lesson_images(ld["title"])
            if pics:
                md = "\n".join(f"* [{p['caption']}]({p['name']})" for p in pics)
                images_blocks.append(f"درس «{ld['title']}» – التصاور:\n{md}\n")
        return "\n".join(images_blocks) if images_blocks else "ما ثـمّـة حتى تصاور."

