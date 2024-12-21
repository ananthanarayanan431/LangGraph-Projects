from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai.chat_models.base import ChatOpenAI
from tools.Internet import Internet_tool
from tools.wikipediaFile import wikipedia_tool
from tools.temperaturee import temperature_tool, get_current_temperature

instructions = """
You are an experienced researcher who always finds high-quality and relevant information on the Internet.
"""

base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

llm = ChatOpenAI(model="gpt-4o",temperature=0.7)
tools = [Internet_tool, wikipedia_tool, temperature_tool]

agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)
query="Please provide information about the BGT cricket match 2024"
result = agent_executor.invoke({"input": query})

print(result['output'])
