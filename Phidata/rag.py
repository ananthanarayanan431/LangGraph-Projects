
from dotenv import load_dotenv
load_dotenv()

from phi.agent.agent import Agent
from phi.model.openai.chat import OpenAIChat
from phi.knowledge.langchain import LangChainKnowledgeBase

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone.vectorstores import PineconeVectorStore

PATH = r"PDF file"

def load_vector_store():
    raw_document = PyPDFLoader(PATH).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
    document = text_splitter.split_documents(raw_document)
    PineconeVectorStore.from_documents(
        documents=document,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        index_name="phidata",
    )

# load_vector_store()
vectorstore = PineconeVectorStore(index_name="phidata",embedding=OpenAIEmbeddings(model="text-embedding-3-large"))
retriever = vectorstore.as_retriever()
knowledge_base = LangChainKnowledgeBase(retriever=retriever)
knowledge_base.load(recreate=False)

actions_agent = Agent(
    name="Action Agent",
    role="Suggest Future action based on last year financial report",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=["Suggest some actions or plans that can be incorporated into the existing plan to increase revenue"],
    knowledge_base=knowledge_base,
    search_knowledge=True,
)

financial_planning_agent = Agent(
    name="Financial Planning Agent",
    role="Provide strategic financial planning based on the last year's financial report",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=["Provide actionable financial planning strategies that align with the company's goals and can optimize resource allocation to increase revenue and profitability."],
    knowledge_base=knowledge_base,
    search_knowledge=True,
)

agent = Agent(
    team=[actions_agent, financial_planning_agent],
    markdown=True,
)

agent.print_response("Give me actionable insights and financial planning", stream=True)
