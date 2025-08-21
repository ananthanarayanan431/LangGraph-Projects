from enum import Enum

from langgraph.graph import END, START


class ScopeNode(str, Enum):
    CLARIFY_WITH_USER = "clarify_with_user"
    WRITE_RESEARCH_BRIEF = "write_research_brief"
    START = START
    END = END


class Conversation(str, Enum):
    messages = "messages"
    research_brief = "research_brief"
    supervisor_messages = "supervisor_messages"
    raw_notes = "raw_notes"
    notes = "notes"
    final_report = "final_report"
