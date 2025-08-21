
from langgraph.graph import START, END

class GraphNode:
    START = START
    END = END
    CLARIFY_WITH_USER = "clarify_with_user"
    WRITE_RESEARCH_BRIEF = "write_research_brief"
    SUPERVISOR_SUBGRAPH = "supervisor_subgraph"
    FINAL_REPORT_GENERATION = "final_report_generation"


class ConversationConfig:
    RESEARCH_BRIEF = "research_brief"
    NOTES = "notes"
    FINAL_REPORT = "final_report"
    MESSAGES = "messages"
    