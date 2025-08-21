
from langgraph.graph import START, END


class GraphNode:
    START = START
    END = END 
    COMPRESSED_RESEARCH = "compressed_research"


class ConversationConfig:
    supervisor_messages = "supervisor_messages"
    research_iterations = "research_iterations"
    research_brief = "research_brief"
    notes = "notes"
    raw_notes = "raw_notes"
    research_topic = "research_topic"


class ToolName:
    SUPERVISOR_TOOLS = "supervisor_tools"
    SUPERVISOR = "supervisor"
    RESEARCH_COMPLETE = "ResearchComplete"
    CONDUCT_RESEARCH = "ConductResearch"
    THINK_TOOL = "think_tool"