
from dotenv import load_dotenv
load_dotenv()

from tools.internet import internet_tool
from tools.temperature import temperature_tool
from tools.wikipedia import wikipedia_tool

from langgraph.prebuilt import create_react_agent
from langgraph_supervisor.supervisor import create_supervisor
from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")


internet_agent = create_react_agent(
    model=llm,
    tools=[internet_tool],
    name="Internet_Expert",
    prompt="""You are a world-class researcher with access to Internet. Use tools to
    Provide a concise and insightful information based on the top search results."""
)

temperature_agent = create_react_agent(
    model=llm,
    tools=[temperature_tool],
    name="Weather_Expert",
    prompt="""You are a world-class weather expert with access to real-time temperature data.
    Use the tools to provide accurate current temperature information for the requested location.
    Ensure your response is concise and insightful"""
)

wikipedia_agent = create_react_agent(
    model=llm,
    tools=[wikipedia_tool],
    name="Wikipedia_Expert",
    prompt="""You are a world-class researcher with access to Wikipedia.
    Use the tools to retrieve and extract relevant information for the given topic.
    Ensure your response is concise, factual, and insightful."""
)

workflow = create_supervisor(
    [internet_agent, wikipedia_agent, temperature_agent],
    model=llm,
    prompt=(
        "You are a team supervisor managing three expert agents: a Wikipedia Expert,an Internet Expert, and a Weather Expert. "
        "Use Wikipedia Expert for general knowledge and historical information. "
        "Use Internet Expert for real-time or recent events. "
        "Use Weather Expert for temperature-related queries."
    )
)

app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "I need the best places to visit in Hyderabad and then give me some result of that place and current temperature of that place"
        }
    ]
})


for message in result['messages']:
    if hasattr(message, 'content'): 
        print(f"Message From: {message.__class__.__name__}")
        print(f"Content: {message.content}")
        if hasattr(message, 'name'):
            print(f"Name: {message.name}")
        if hasattr(message, 'tool_calls') and len(message.tool_calls)>1:
            print(f"Tool Calls: {message.tool_calls[0]['name']}")
        print("-" * 40)
