
from dotenv import load_dotenv

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from graph.chains.model import GradeDocuments

load_dotenv()


llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

sytem_prompt = """
You are a grader assessing relevance of a retrieved document to a user question. \n 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.

Question: {question}
Document: {document}
"""

grade_prompt = ChatPromptTemplate.from_template(sytem_prompt)

retrieval_grader = grade_prompt | structured_llm_grader
