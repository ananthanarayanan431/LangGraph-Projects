
from typing import Any, Dict

from graph.chains.generation import generation_cahin
from graph.state import AgentState


def generate(state: AgentState)->Dict[str, Any]:
    """Generate a response for the agent"""

    question = state['question']
    documents = state['documents']
    generation = generation_cahin.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
