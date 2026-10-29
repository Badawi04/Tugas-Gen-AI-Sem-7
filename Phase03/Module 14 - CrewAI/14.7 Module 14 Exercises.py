# 14.7 Module 14 Exercises (Halaman 163)
"""Content pipeline, code review crew, custom search tool, dan process comparison."""

import os
import time
from pathlib import Path
from typing import Type

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(PROJECT_ENV, override=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def create_llm() -> LLM:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY belum tersedia di file .env proyek."
        )
    return LLM(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
    )


class MockSearchInput(BaseModel):
    query: str = Field(description="Search query string")


class WebSearchTool(BaseTool):
    name: str = "Mock Web Search"
    description: str = (
        "Search predefined technical sources for a topic. "
        "Use this when factual research is needed."
    )
    args_schema: Type[BaseModel] = MockSearchInput

    def _run(self, query: str) -> str:
        results = {
            "fine-tuning vs rag": (
                "Fine-tuning changes model weights for a task or style. "
                "RAG retrieves current external context at inference time. "
                "Fine-tuning is useful for behavior; RAG is useful for changing facts."
            ),
            "rag": (
                "RAG combines retrieval with generation. Production concerns "
                "include chunking, retrieval quality, citations, and latency."
            ),
            "fine-tuning": (
                "Fine-tuning adapts a pretrained model with task-specific examples. "
                "It requires training data and can make updates more expensive."
            ),
        }
        normalized = query.lower()
        for key, result in results.items():
            if key in normalized or all(word in normalized for word in key.split()):
                return f"Search results for '{query}': {result}"
        return (
            f"Search results for '{query}': compare primary sources, "
            "check dates, and separate retrieved facts from model assumptions."
        )


def make_agent(role: str, goal: str, backstory: str, tools=None, **kwargs) -> Agent:
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=create_llm(),
        tools=tools or [],
        verbose=False,
        max_iter=5,
        **kwargs,
    )


def build_blog_crew() -> Crew:
    researcher = make_agent(
        "Researcher",
        "Find accurate, practical points for a technical blog post.",
        "You verify claims and distinguish RAG from fine-tuning.",
        tools=[WebSearchTool()],
    )
    outline_writer = make_agent(
        "Outline Writer",
        "Turn research into a logical article structure.",
        "You design clear outlines for senior developers.",
    )
    full_writer = make_agent(
        "Full Writer",
        "Expand every outline section into a useful technical article.",
        "You write precise, practical developer documentation.",
    )
    research_task = Task(
        description=(
            "Research the topic 'The practical guide to fine-tuning vs RAG'. "
            "Use the web search tool and produce key points, trade-offs, "
            "and recommendations."
        ),
        expected_output="A factual list of key points and trade-offs.",
        agent=researcher,
    )
    outline_task = Task(
        description=(
            "Create a numbered outline for the blog post from the research. "
            "Include introduction, comparison, decision guide, and conclusion."
        ),
        expected_output="A numbered outline with section goals.",
        agent=outline_writer,
        context=[research_task],
    )
    writing_task = Task(
        description=(
            "Write the complete blog post from the research and outline. "
            "Use Markdown headings, concrete examples, and a final recommendation."
        ),
        expected_output="A complete technical blog post in Markdown.",
        agent=full_writer,
        context=[research_task, outline_task],
    )
    return Crew(
        agents=[researcher, outline_writer, full_writer],
        tasks=[research_task, outline_task, writing_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )


def build_code_review_crew(code: str) -> Crew:
    developer = make_agent(
        "Developer",
        "Explain and improve the supplied Python code.",
        "You write maintainable Python and identify assumptions.",
    )
    security = make_agent(
        "Security Reviewer",
        "Find vulnerabilities and unsafe input handling.",
        "You focus on injection, secrets, access control, and unsafe execution.",
    )
    performance = make_agent(
        "Performance Reviewer",
        "Find inefficiencies and scalability problems.",
        "You focus on algorithmic complexity, I/O, and resource usage.",
    )
    lead = make_agent(
        "Lead Reviewer",
        "Merge all feedback into one prioritized final report.",
        "You produce actionable engineering review reports.",
    )
    implementation = Task(
        description=f"Review this Python snippet and describe its behavior:\n```python\n{code}\n```",
        expected_output="A concise implementation assessment.",
        agent=developer,
    )
    security_task = Task(
        description="Review the code for vulnerabilities and explain severity and fixes.",
        expected_output="A security findings list with severity and remediation.",
        agent=security,
        context=[implementation],
    )
    performance_task = Task(
        description="Review the code for inefficiency and propose concrete improvements.",
        expected_output="A performance findings list with complexity and fixes.",
        agent=performance,
        context=[implementation],
    )
    lead_task = Task(
        description="Merge implementation, security, and performance feedback into a final report.",
        expected_output="Prioritized final code review report.",
        agent=lead,
        context=[implementation, security_task, performance_task],
    )
    return Crew(
        agents=[developer, security, performance, lead],
        tasks=[implementation, security_task, performance_task, lead_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )


def run_search_tool_check() -> str:
    tool = WebSearchTool()
    return tool.run("fine-tuning vs RAG")


def _three_task_crew(process: Process) -> Crew:
    first = make_agent("Researcher", "Research the topic.", "You find technical facts.")
    second = make_agent("Analyst", "Analyse the research.", "You compare evidence.")
    writer = make_agent("Writer", "Write a concise answer.", "You communicate clearly.")
    tasks = [
        Task(
            description="Research vector databases for RAG.",
            expected_output="Five factual research points.",
            agent=first,
        ),
        Task(
            description="Analyse the research and identify three recommendations.",
            expected_output="Three practical recommendations.",
            agent=second,
        ),
        Task(
            description="Write a final technical summary from the earlier work.",
            expected_output="A structured summary with recommendations.",
            agent=writer,
        ),
    ]
    if process == Process.hierarchical:
        manager = make_agent(
            "Project Manager",
            "Coordinate workers and ensure the final summary is complete.",
            "You delegate work and check quality.",
            allow_delegation=True,
        )
        return Crew(
            agents=[first, second, writer],
            tasks=tasks,
            process=process,
            manager_agent=manager,
            verbose=False,
        )
    return Crew(
        agents=[first, second, writer],
        tasks=tasks,
        process=process,
        verbose=False,
        memory=False,
    )


def compare_processes() -> dict[str, dict]:
    comparison: dict[str, dict] = {}
    for process in (Process.sequential, Process.hierarchical):
        started = time.perf_counter()
        result = _three_task_crew(process).kickoff()
        elapsed = time.perf_counter() - started
        output = str(result)
        quality_markers = sum(
            marker in output.lower()
            for marker in ("vector", "rag", "recommend", "retriev")
        )
        comparison[process.value] = {
            "execution_time_seconds": round(elapsed, 2),
            "quality_marker_count": quality_markers,
            "output": output,
        }
    return comparison


if __name__ == "__main__":
    blog_result = build_blog_crew().kickoff()
    print("=== FINAL BLOG POST ===")
    print(blog_result)

    sample_code = """
def find_user(users, name):
    for user in users:
        if user["name"] == name:
            return user
    return None
"""
    print("\n=== CODE REVIEW REPORT ===")
    print(build_code_review_crew(sample_code).kickoff())

    print("\n=== CUSTOM WEB SEARCH TOOL ===")
    print(run_search_tool_check())

    print("\n=== SEQUENTIAL VS HIERARCHICAL ===")
    for process_name, result in compare_processes().items():
        print(
            process_name,
            "time=",
            result["execution_time_seconds"],
            "quality_markers=",
            result["quality_marker_count"],
        )
