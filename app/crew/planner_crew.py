from typing import ClassVar
from pathlib import Path
from crewai import Agent, Crew, Task, LLM, Process
from crewai.project import CrewBase, agent, crew, task, llm, tool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
import google.generativeai as genai
from app.crew.tools import LessonRetrieverTool
from app.crew.knowledge_graph import Neo4jKG
from app.crew.config import embedder_cfg
from app.handlers import get_sessions_logs, get_user_logs
import json
from app.crew.config import URI, USER, PASSWORD


@CrewBase
class PlannerCrew: 
    base_directory: ClassVar[Path] = Path(__file__).parent
    agents_config: ClassVar[str] = 'config/agents.yaml'
    tasks_config:  ClassVar[str] = 'config/tasks.yaml'

    @llm
    def llm_cfg(self) -> LLM:
        return LLM(model="gemini/gemini-2.5-flash-lite", temperature=0.5, max_tokens=4000)

    @tool
    def lesson_retriever_tool(self) -> LessonRetrieverTool:
        kg = Neo4jKG(URI, USER, PASSWORD)
        return LessonRetrieverTool(kg=kg)

    @agent
    def sessions_history_agent(self) -> Agent:
        session_logs_input = get_sessions_logs()
        json_str = json.dumps(session_logs_input, ensure_ascii=False, indent=2)
        knowledge_source = StringKnowledgeSource(content=json_str)
        return Agent(
            config=self.agents_config['sessions_history_agent'],
            knowledge_sources=[knowledge_source],
            embedder=embedder_cfg,
        )

    @agent
    def user_history_agent(self) -> Agent:
        user_logs_input = get_user_logs()
        json_str = json.dumps(user_logs_input, ensure_ascii=False, indent=2)
        knowledge_source = StringKnowledgeSource(content=json_str)
        return Agent(
            config=self.agents_config['user_history_agent'],
            knowledge_sources=[knowledge_source],
            embedder=embedder_cfg,
        )

    @agent
    def planner_agent(self) -> Agent:
        return Agent(config=self.agents_config['planner_agent'])

    @task
    def user_history_task(self) -> Task:
        return Task(
            config=self.tasks_config['user_history_task'],
            agent=self.user_history_agent(),
        )

    @task
    def sessions_history_task(self) -> Task:
        return Task(
            config=self.tasks_config['sessions_history_task'],
            agent=self.sessions_history_agent(),
        )

    @task
    def plan_task(self) -> Task:
        return Task(
            config=self.tasks_config['plan_task'],
            agent=self.planner_agent(),
            process=Process.sequential
        )

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)
