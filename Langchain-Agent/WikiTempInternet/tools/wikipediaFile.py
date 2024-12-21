
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field

class WikiInputs(BaseModel):
    """Inputs to the Wikipedia tool"""
    query:str = Field(description="Query to look up in wikipedia, should 3 or less words")


@tool
def wikipedia_tool(query:str):
    """Inputs to the Wikipedia tool"""
    wrapper = WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=2000)
    tool = WikipediaQueryRun(
        name="Wiki-Input",
        api_wrapper=wrapper,
        args_schema=WikiInputs,
        return_direct=True,
    )
    return tool.run({"query": query})
