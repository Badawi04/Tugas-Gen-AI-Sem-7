# 9.5 Output Parsers (Halaman 108-109)
import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY belum tersedia di file .env")

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    temperature=0.7,
    api_key=api_key,
)

# Pydantic schema for structured output
class ModelComparison(BaseModel):
    models: List[str] = Field(description="Model names compared")
    best_for_cost: str = Field(description="Most cost-effective model")
    best_for_quality: str = Field(description="Highest quality model")
    recommendation: str = Field(description="Overall recommendation")

# Gunakan JsonOutputParser (kompatibel dengan semua model)
parser = JsonOutputParser(pydantic_object=ModelComparison)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI model comparison expert. {format_instructions}"),
    ("human", "Compare these models for a startup on a budget: {models}"),
])

chain = prompt | llm | parser

if __name__ == "__main__":
    result = chain.invoke({
        "models": "gpt-4o, gpt-4o-mini, claude-sonnet-4-5",
        "format_instructions": parser.get_format_instructions()
    })
    print(f"Best for cost   : {result['best_for_cost']}")
    print(f"Best for quality: {result['best_for_quality']}")
    print(f"Recommendation  : {result['recommendation']}")
