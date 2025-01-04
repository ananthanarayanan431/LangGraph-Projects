
from dotenv import load_dotenv
load_dotenv()

import os
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="langgraph-project"

from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages.base import BaseMessage
import operator

from langchain_core.tools import tool, StructuredTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai.chat_models.base import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, END, StateGraph
from langchain_core.agents import AgentFinish

@tool("WikipediaTool",return_direct=True)
def wikipedia(query:str):
    """Useful for searching in Wikipedia"""
    wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
    tools = WikipediaQueryRun(api_wrapper=wrapper)
    return tools.run({"query": query})

@tool("WeatherTool",return_direct=True)
def temperature(query:str):
    """Useful for searching Weather of particular location"""
    weather = OpenWeatherMapAPIWrapper()
    return weather.run(query)

@tool("InternetTool",return_direct=True)
def Internet(query:str):
    """Useful for Searching in Internet"""
    tavily_tool = TavilySearchResults(api_wrapper=TavilySearchAPIWrapper(),max_results=5)
    return tavily_tool.run({"query": query})

class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


tools=[wikipedia, temperature, Internet]
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model_name="gpt-4o-mini",temperature=0.7)
agent_runnable = create_openai_functions_agent(llm,
                                               tools,
                                               prompt)

tools_executor = ToolNode(tools)

def run_agent(input):
    agent_outcome = agent_runnable.invoke(input)
    return {"agent_outcome": agent_outcome}

def execute_tools(input):
    agent_action = input['agent_outcome']
    output = tools_executor.invoke(agent_action)
    print(f"The agent action is {agent_action}")
    print(f"The tool result is: {output}")
    return {"intermediate_steps": [(agent_action, str(output))]}


def should_continue(input):
    if isinstance(input['agent_outcome'],AgentFinish):
        return "end"
    return "continue"

workflow = StateGraph(AgentState)
workflow.add_node("agent",run_agent)
workflow.add_node("action",execute_tools)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)
workflow.add_edge("action","agent")

app = workflow.compile()

inputs = {"input": "give me a random state in India and give me wikipedia result of that state"}
for s in app.stream(inputs):
    print(list(s.values())[0])
    print("----"*10)
