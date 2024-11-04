
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph import START,END
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm=ChatOpenAI(model="gpt-4o-mini")

class State(TypedDict):
    messages: Annotated[list,add_messages]

def chatBot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph=StateGraph(State)
graph.add_node("chatbot",chatBot)
# graph.set_entry_point("chatbot")
# graph.set_finish_point("chatbot")

graph.add_edge(START,'chatbot')
graph.add_edge('chatbot',END)

graph1 = graph.compile()

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        break
    for event in graph1.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
