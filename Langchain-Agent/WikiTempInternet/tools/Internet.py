
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field


class InternetTool(BaseModel):
    """Input for the Tavily Tool"""
    query: str = Field(description="search query to look up")

@tool
def Internet_tool(query:str):
    """Input for the Tavily Tool"""
    wrapper = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(
        name="Internet-Tool",
        api_wrapper=wrapper,
        max_results=5,
        args_schema=InternetTool,
    )
    return tavily_tool.run({"query": query})
