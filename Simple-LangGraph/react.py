
from dotenv import load_dotenv
load_dotenv()

from langchain_openai.chat_models import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import MemorySaver

def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.
    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.
    Args:
        a: first int
        b: second int
    """
    return a / b

search = DuckDuckGoSearchRun()
tools = [multiply, add, divide, search]

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
llm_with_tools = llm.bind_tools(tools=tools)

sys_msg = SystemMessage(content="You are a helpful assistant tasked with using search and performing arithmetic on a set of inputs.")

def reasoner(state:MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state['messages'])]}


builder = StateGraph(MessagesState)
builder.add_node("reasoner",reasoner)
builder.add_node("tools",ToolNode(tools))

builder.add_edge(START,"reasoner")
builder.add_conditional_edges(
    "reasoner",
    tools_condition
)
builder.add_edge("tools","reasoner")

graph = builder.compile(checkpointer=MemorySaver())
messages = [HumanMessage(content="What is 2 times of Virat Kohli's age?")]
answer = graph.invoke({"messages": messages})

for m in answer['messages']:
    m.pretty_print()
