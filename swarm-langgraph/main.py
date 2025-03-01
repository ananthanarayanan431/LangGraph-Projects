
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm

from tools.internet import internet_tool
from tools.temperature import temperature_tool
from tools.wikipedia import wikipedia_tool

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

transfers_to_internet = create_handoff_tool(
    agent_name="internet_assistance",
    description="Transfer to Wikipedia_assistance, This tool may help you to relevant information about particular place/person"
)

transfers_to_wikipedia = create_handoff_tool(
    agent_name="Wikipedia_assistance",
    description="Transfer to internet_assistance, This Tool may help you to relevant information from internet"
)

transfers_to_temperature = create_handoff_tool(agent_name="temperature_assistant")

temperature_assistant = create_react_agent(
    llm,
    [temperature_tool,transfers_to_internet],
    prompt="You are temperature_assistant, a temperature expert. You provide accurate temperature readings, conversions (Celsius, Fahrenheit, Kelvin), and insights on weather and climate",
    name="temperature_assistant",
)

internet_assistance = create_react_agent(
    llm,
    [internet_tool,transfers_to_wikipedia,transfers_to_temperature],
    prompt="You are internet_assistance, an internet expert. You fetch real-time web data, answer queries with up-to-date information, and provide accurate online insights.",
    name="internet_assistance"
)

Wikipedia_assistance = create_react_agent(
    llm,
    [wikipedia_tool,transfers_to_internet],
    prompt="You are Wikipedia_assistance, a Wikipedia expert. You provide accurate, summarized, and well-cited information from Wikipedia on any topic",
    name="Wikipedia_assistance",
)

builder = create_swarm(
    [temperature_assistant, internet_assistance, Wikipedia_assistance],
    default_active_agent="internet_assistance",
)

app = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "1"}}

response = app.invoke(
    {"messages": [{"role": "user", "content": "I want to know climate of Hyderabad"}]},
    config,
)

print(response)
