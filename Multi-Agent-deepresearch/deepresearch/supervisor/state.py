

import operator
from typing import Annotated, Sequence
from langgraph.graph import MessagesState
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class SupervisorState(MessagesState):
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: str
    notes: Annotated[list[str], operator.add]
    research_iterations: int = 0
    raw_notes: Annotated[list[str], operator.add]