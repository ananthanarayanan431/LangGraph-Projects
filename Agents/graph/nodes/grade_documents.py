
from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import AgentState


def grade_documents(state: AgentState)->Dict[str, Any]:
    """Grade the documents for the agent"""

    question = state['question']
    documents = state['documents']
    websearch = False 

    filtered_documents = []
    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc})
        grade = score.binary_score
        if grade.lower() == "yes":
            filtered_documents.append(doc)
        else:
            websearch = True

    return {"documents": filtered_documents, "web_search": websearch, 'question': question}
