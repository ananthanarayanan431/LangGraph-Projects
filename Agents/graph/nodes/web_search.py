from json import load
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import AgentState

load_dotenv()
web_search_tool = TavilySearch(max_results=5)

def web_search(state: AgentState)->Dict[str, Any]:
    """Web search for the agent"""
    question = state["question"]
    documents = state.get("documents", [])
        
    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    documents.append(web_results)
    return {"documents": documents, "question": question, "web_search": True}
