
from typing import TypedDict, Annotated
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq.chat_models import ChatGroq
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

arxiv_wrapper=ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=300)
arxiv_tool=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=300)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)

tools=[arxiv_tool,wiki_tool]
# llm=ChatGroq(temperature=0.7,model="llama-3.1-70b-versatile")
llm=ChatOpenAI(temperature=0.7,model="gpt-4o-mini")

llm_tools = llm.bind_tools(tools=tools)

class State(TypedDict):
    messages: Annotated[list,add_messages]

def chatbot(state:State):
  return {"messages":[llm_tools.invoke(state["messages"])]}

graph = StateGraph(State)
graph.add_node("chatbot",chatbot)
graph.add_edge(START,"chatbot")

tool_node=ToolNode(tools=tools)
graph.add_node("tools",tool_node)

graph.add_conditional_edges(
    "chatbot",
    tools_condition
)
graph.add_edge("tools","chatbot")
graph.add_edge("chatbot",END)

graph1 = graph.compile()
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        break
    for event in graph1.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
