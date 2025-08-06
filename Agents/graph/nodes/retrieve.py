from typing import Any, Dict

from graph.state import AgentState
from ingestion import retriever

def retrieve(state: AgentState)->Dict[str, Any]:
    """Retrieve the documents for the agent"""

    question = state['question']
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question, "web_search": False}