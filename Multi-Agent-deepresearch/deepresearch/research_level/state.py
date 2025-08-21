
import operator
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, List, TypedDict


class ResearcherState(MessagesState):
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]


class ResearcherOutputState(MessagesState):
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]