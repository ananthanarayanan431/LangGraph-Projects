
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages.human import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph.state import StateGraph, START, END

class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model_name="gpt-4o",temperature=0.7)

def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    return a * b

llm_with_tools = llm.bind_tools([multiply])

def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_edge(START,"tool_calling_llm")
builder.add_edge("tool_calling_llm",END)

graph = builder.compile()
messages = graph.invoke({"messages": HumanMessage(content="Hello!")})
for m in messages['messages']:
    m.pretty_print()
