from neo4j import GraphDatabase
from typing import Dict, Tuple

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

def _ask_user_for_topic(kg: Neo4jKG) -> str | None:
    """
    Prints a numbered list of all topics in KG,
    asks the user to select one by typing its number or exact name.
    Returns the topic name (Arabic) or None if aborted.
    """
    topics = kg.list_all_topics()
    print("\n📚 Available topics in the KG:")
    for idx, tname in enumerate(topics, start=1):
        print(f"  {idx}) {tname}")
    selection = input("\n🔢 Enter the number or exact name of the topic (or 'خروج' to cancel): ").strip()
    if selection.lower() == "خروج":
        return None
    # If user typed a number, convert to index
    if selection.isdigit():
        idx = int(selection) - 1
        if 0 <= idx < len(topics):
            return topics[idx]
        else:
            print("⚠️ Invalid number. Try again.")
            return _ask_user_for_topic(kg)
    # Otherwise, see if the typed string exactly matches a topic
    if selection in topics:
        return selection
    print("⚠️ That topic was not found. Please choose again.")
    return _ask_user_for_topic(kg)

def _infer_topic_from_question(question: str, kg: Neo4jKG) -> str | None:
    """
    Try to guess which topic the question refers to by checking if any topic name
    or lesson title appears as a substring. If none, return None.
    """
    # 1) First check topic names
    all_topics = kg.list_all_topics()
    for tname in all_topics:
        if tname in question:
            return tname

    # 2) If no topic found, check each lesson title
    for tname in all_topics:
        lessons = kg.get_lessons_for_topic(tname)
        for ld in lessons:
            if ld["title"] in question:
                return tname

    return None
