
from smolagents import CodeAgent
from smolagents import HfApiModel

from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool

load_dotenv()

def wikipedia(query:str):
    wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
    tools = WikipediaQueryRun(api_wrapper=wrapper, return_direct=True)
    return tools.run({"query": query})

def temperature(query:str):
    weather = OpenWeatherMapAPIWrapper()
    return weather.run(query)

def Internet(query:str):
    tavily_tool = TavilySearchResults(api_wrapper=TavilySearchAPIWrapper(),max_results=5)
    return tavily_tool.run({"query": query})


wikipedia_tool = Tool.from_function(
    func=wikipedia,
    name="Wikipedia Answers",
    description="Query to look up in wikipedia"
)

temperature_tool = Tool.from_function(
    func=temperature,
    name="Temperature Function",
    description="Fetch current temperature for a given place"
)

internet_tool = Tool.from_function(
    func=Internet,
    name="Internet Search Function",
    description="Search query to look up"
)

model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")
agent = CodeAgent(
    tools=[internet_tool,temperature_tool, wikipedia_tool],
    model=model,
)
print("***Agent Started Working***")
print(agent.run("Give me New year wishes for software Engineer"))
print("***Agent completed work***")
