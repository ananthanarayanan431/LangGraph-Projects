
"""state of the Graph"""

from typing import List, Annotated
from langgraph.graph import MessagesState
from langchain_core.documents import Document


class AgentState(MessagesState):
    question: str 
    generation: str 
    web_search: bool 
    documents: List[Document]