
from enum import Enum as PyEnum
from langgraph.graph import START, END

class ResearchNode(str, PyEnum):
    START = START
    END = END
    LLM_CALL = "llm_call"
    TOOL_NODE = "tool_node"
    COMPRESS_RESEARCH = "compress_research"

class ConversationResearch(str, PyEnum):
    researcher_messages = "researcher_messages"
    tool_call_iterations = "tool_call_iterations"
    research_topic = "research_topic"
    compressed_research = "compressed_research"
    raw_notes = "raw_notes"
