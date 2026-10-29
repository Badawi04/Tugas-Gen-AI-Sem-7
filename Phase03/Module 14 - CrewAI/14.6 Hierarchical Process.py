# 14.6 Hierarchical Process with a Manager (Halaman 161-163)
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    llm = "openai/gpt-4o-mini"

    manager = Agent(
        role="AI Content Production Manager",
        goal="Coordinate a team to produce high-quality AI content efficiently.",
        backstory="""You are an experienced content production manager who excels at
breaking down projects, assigning work to the right specialist, and ensuring
quality standards are met before publication.""",
        llm=llm,
        allow_delegation=True,
        verbose=False,
    )
    researcher = Agent(
        role="AI Research Analyst",
        goal="Research AI topics thoroughly.",
        backstory="Senior researcher specialising in LLM systems and RAG.",
        llm=llm,
        verbose=False,
    )
    writer = Agent(
        role="Technical Writer",
        goal="Write developer-focused technical content.",
        backstory="Technical writer with Python and ML experience.",
        llm=llm,
        verbose=False,
    )

    main_task = Task(
        description="""Produce a complete technical brief on 'Vector Databases for RAG in Production'.
Include: what they are, how to choose one, and three implementation best practices.""",
        expected_output="A structured technical brief in Markdown, 400-500 words.",
        agent=manager,
    )

    crew = Crew(
        # The manager is configured separately and must not be a worker agent.
        agents=[researcher, writer],
        tasks=[main_task],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=False,
    )
    print("Hierarchical Manager-led Crew compiled successfully.")
