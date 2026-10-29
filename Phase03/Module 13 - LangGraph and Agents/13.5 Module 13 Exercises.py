# 13.5 Module 13 Exercises (Halaman 152)
"""Research agent, ReAct step limit, Plan-and-Execute, dan memory multi-turn."""

import operator
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

load_dotenv()


def create_llm(temperature: float = 0) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY belum tersedia. Tambahkan key tersebut ke file .env."
        )
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# 1. Research agent dengan web_search, summarise, dan save_note
# ---------------------------------------------------------------------------
SAVED_NOTES: list[dict[str, str]] = []


def web_search(topic: str) -> str:
    """Mock web search yang mengembalikan hasil tetap untuk latihan."""
    return (
        f"Finding A about {topic}: it improves access to relevant information.\n"
        f"Finding B about {topic}: evaluation and reliable sources are important.\n"
        f"Finding C about {topic}: practical systems need clear objectives."
    )


def summarise(text: str) -> str:
    """Ringkas hasil penelitian menggunakan LLM."""
    llm = create_llm()
    response = llm.invoke(
        [
            SystemMessage(content="Summarise the research in three concise sentences."),
            HumanMessage(content=text),
        ]
    )
    return str(response.content)


def save_note(content: str) -> str:
    """Simpan catatan terstruktur ke list in-memory."""
    note = {"title": "Research note", "content": content}
    SAVED_NOTES.append(note)
    return f"Note saved successfully. Total notes: {len(SAVED_NOTES)}"


RESEARCH_TOOLS = {
    "web_search": web_search,
    "summarise": summarise,
    "save_note": save_note,
}

RESEARCH_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for a topic using the mock search tool.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarise",
            "description": "Summarise research findings.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a structured research note.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    },
]


class ResearchState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_calls: int


def research_agent_node(state: ResearchState) -> dict:
    llm = create_llm().bind_tools(RESEARCH_TOOL_SCHEMAS)
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Research the user's topic. Use web_search first, then "
                    "summarise the findings, then save_note with a structured note."
                )
            )
        ]
        + state["messages"]
    )
    return {"messages": [response]}


def research_tools_node(state: ResearchState) -> dict:
    last_message = state["messages"][-1]
    outputs = []
    for tool_call in last_message.tool_calls:
        tool = RESEARCH_TOOLS.get(tool_call["name"])
        if tool is None:
            raise ValueError(f"Unknown research tool: {tool_call['name']}")
        outputs.append(
            ToolMessage(
                content=str(tool(**tool_call["args"])),
                tool_call_id=tool_call["id"],
            )
        )
    return {
        "messages": outputs,
        "tool_calls": state["tool_calls"] + len(outputs),
    }


def research_route(state: ResearchState) -> str:
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else END


def step_limit_node(_: ResearchState) -> dict:
    return {"messages": [AIMessage(content="Reached step limit.")]}


def build_research_agent(max_tool_calls: int = 8):
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls harus lebih besar dari 0.")

    graph = StateGraph(ResearchState)
    graph.add_node("agent", research_agent_node)
    graph.add_node("tools", research_tools_node)
    graph.add_node("limit", step_limit_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        research_route,
        {"tools": "tools", END: END},
    )

    def after_tools(state: ResearchState) -> str:
        return "limit" if state["tool_calls"] >= max_tool_calls else "agent"

    graph.add_conditional_edges(
        "tools",
        after_tools,
        {"agent": "agent", "limit": "limit"},
    )
    graph.add_edge("limit", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# 2. Plan-and-Execute dengan planner dan executor
# ---------------------------------------------------------------------------
class PlanState(TypedDict):
    topic: str
    plan: list[str]
    current_step: int
    results: Annotated[list[str], operator.add]
    answer: str


def planner_node(state: PlanState) -> dict:
    response = create_llm().invoke(
        "Create exactly three numbered research steps for this topic: "
        f"{state['topic']}"
    )
    steps = [
        line.strip()
        for line in str(response.content).splitlines()
        if line.strip()
    ][:3]
    return {"plan": steps, "current_step": 0}


def executor_node(state: PlanState) -> dict:
    if state["current_step"] >= len(state["plan"]):
        return {"answer": "\n".join(state["results"])}

    step = state["plan"][state["current_step"]]
    finding = web_search(step)
    next_step = state["current_step"] + 1
    return {
        "results": [f"Step {state['current_step'] + 1}: {finding}"],
        "current_step": next_step,
        "answer": finding if next_step == len(state["plan"]) else "",
    }


def plan_route(state: PlanState) -> str:
    return "executor" if state["current_step"] < len(state["plan"]) else END


def build_plan_and_execute():
    graph = StateGraph(PlanState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        plan_route,
        {"executor": "executor", END: END},
    )
    graph.add_conditional_edges(
        "executor",
        plan_route,
        {"executor": "executor", END: END},
    )
    return graph.compile()


# ---------------------------------------------------------------------------
# 3. Multi-turn agent dengan MemorySaver dan thread_id tetap
# ---------------------------------------------------------------------------
class ConversationState(TypedDict):
    messages: Annotated[list, operator.add]


def conversation_node(state: ConversationState) -> dict:
    response = create_llm().invoke(
        [
            SystemMessage(
                content="Answer using the conversation history. Remember prior context."
            )
        ]
        + state["messages"]
    )
    return {"messages": [response]}


def build_memory_agent():
    graph = StateGraph(ConversationState)
    graph.add_node("conversation", conversation_node)
    graph.add_edge(START, "conversation")
    graph.add_edge("conversation", END)
    return graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    research_agent = build_research_agent(max_tool_calls=8)
    research_result = research_agent.invoke(
        {
            "messages": [
                HumanMessage(content="Research retrieval-augmented generation.")
            ],
            "tool_calls": 0,
        }
    )
    print("Research result:", research_result["messages"][-1].content)
    print("Saved notes:", SAVED_NOTES)

    planner = build_plan_and_execute()
    plan_result = planner.invoke(
        {
            "topic": "vector databases",
            "plan": [],
            "current_step": 0,
            "results": [],
            "answer": "",
        }
    )
    print("\nPlan-and-Execute:", plan_result["results"])

    memory_agent = build_memory_agent()
    memory_config = {"configurable": {"thread_id": "student-session-1"}}
    first = memory_agent.invoke(
        {"messages": [HumanMessage(content="My project topic is RAG.")]},
        config=memory_config,
    )
    second = memory_agent.invoke(
        {"messages": [HumanMessage(content="What is my project topic?")]},
        config=memory_config,
    )
    print("\nFirst:", first["messages"][-1].content)
    print("Second:", second["messages"][-1].content)
