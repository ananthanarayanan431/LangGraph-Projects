
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai.chat_models import ChatOpenAI

from graph.chains.model import GradeHallucination

llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucination)

system_prompt = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.

Document: {document}
Generation: {generation}
"""

hallucination_prompt = ChatPromptTemplate.from_template(system_prompt)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader