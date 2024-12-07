
from dotenv import load_dotenv
load_dotenv()

import requests, os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai.chat_models import ChatOpenAI
from langchain_groq.chat_models import ChatGroq
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

URL = "https://r.jina.ai/<LINK>"

LLM = ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
LLM_FAST = ChatGroq(model="llama-3.1-8b-instant")

PLACE = "./PLACE"

headers = {'Authorization' : 'Bearer ' + os.environ['JINA_API_KEY']}
response = requests.get(url=URL,headers=headers)

if response.status_code==200:
    content=response.text
else:
    raise Exception(f"Failed to retrieve data {response.status_code}")

text_splitters = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)

splits = text_splitters.split_text(content)
documents = [Document(page_content=text) for text in splits]

if not os.path.exists(PLACE):
    vectorstore = Chroma.from_documents(documents=documents,embedding=OpenAIEmbeddings(),persist_directory=PLACE)
else:
    vectorstore = Chroma(persist_directory=PLACE,embedding_function=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search,max_results=5)

retrieval_tool = create_retriever_tool(
    retriever,
    "<NAME> Information",
    "Useful for general information about <NAME> company policy and other regulation"
)

tools = [tavily_tool, retrieval_tool]
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(LLM, tools, prompt)
agent_executor = AgentExecutor(agent=agent,tools=tools)

while True:
    user_input = input("Hey, Enter your question: ")
    if user_input.lower() in ['quit','exit','nothing','q']:
        break
    else:
        print("BOT: ",agent_executor.invoke({"input": user_input})['output'])
