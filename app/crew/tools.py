from crewai.tools import BaseTool
import os
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import Type, List, Any
from app.crew.knowledge_graph import Neo4jKG

class LessonRetrievalInput(BaseModel):
    topic_name: str

class LessonRetrieverTool(BaseTool):
    name: str = "get_lessons_for_topic"
    description: str = (
        "يسترجع قائمة بالدروس (بعناوينها وأرقام الصفحات) المتعلقة بموضوع معين في قاعدة المعرفة."
        " أدخل اسم الموضوع بدقة مثل 'Fractions' أو 'Electricity'."
    )
    args_schema: Type[BaseModel] = LessonRetrievalInput

    def __init__(self, kg: Neo4jKG):
        super().__init__()
        object.__setattr__(self, "_kg", kg)

    def _run(self, topic_name: str, **kwargs: Any) -> List[dict]:
        print(f"fetching lessons for topic: {topic_name}")
        return self._kg.get_lessons_for_topic(topic_name)

    def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This tool does not support async execution")
