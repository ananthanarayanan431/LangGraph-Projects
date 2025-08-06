

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

from graph.chains.model import RouterQuery

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)

structured_llm = llm.with_structured_output(RouterQuery)

system_prompt = """
You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search.

{question}
"""

route_prompt = ChatPromptTemplate.from_template(system_prompt)

question_router = route_prompt | structured_llm
