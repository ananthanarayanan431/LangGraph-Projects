
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai.chat_models import ChatOpenAI

from graph.chains.model import GradeAnswer


llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system_prompt = """
You are a grader assessing whether an answer addresses / resolves a question \n 
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.

Document: {document}
Generation: {generation}
"""

answer_prompt = ChatPromptTemplate.from_template(system_prompt)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
