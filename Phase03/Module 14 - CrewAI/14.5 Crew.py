# 14.5 Crew - Putting It Together (Halaman 159-161)
import importlib.util
from pathlib import Path

from crewai import Agent, Crew, Process
from dotenv import load_dotenv

load_dotenv()


def load_module(filename: str, module_name: str):
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Tidak dapat memuat modul: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    tools_module = load_module("14.3 Tools.py", "module_14_3_tools")
    tasks_module = load_module("14.4 Tasks.py", "module_14_4_tasks")
    cost_tool = tools_module.cost_tool
    context_tool = tools_module.context_tool
    make_tasks = tasks_module.make_tasks

    llm = "openai/gpt-4o-mini"

    researcher = Agent(
        role="AI Research Analyst",
        goal="Find accurate information on AI topics.",
        backstory="Senior AI research analyst with expertise in LLM systems.",
        llm=llm,
        tools=[context_tool],
        verbose=False,
        max_iter=5,
    )
    writer = Agent(
        role="Technical Content Writer",
        goal="Write clear technical content for developers.",
        backstory="Technical writer with a software engineering background.",
        llm=llm,
        tools=[cost_tool],
        verbose=False,
        max_iter=5,
    )
    reviewer = Agent(
        role="Content Quality Reviewer",
        goal="Ensure content is accurate and publication-ready.",
        backstory="Meticulous editor who reviews technical AI content.",
        llm=llm,
        verbose=False,
        max_iter=5,
    )

    research_task, writing_task, review_task = make_tasks(researcher, writer, reviewer)

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )

    print("Sequential Crew assembled successfully.")
