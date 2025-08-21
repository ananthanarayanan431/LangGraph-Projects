"""Scope level Graph"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langgraph.graph import StateGraph
from langgraph.types import Command

from deepresearch.scope_level.constant import Conversation, ScopeNode
from deepresearch.scope_level.model import ClarifyWithUser, ResearchQuestion
from deepresearch.scope_level.prompts import (
    CLARIFY_USER_INSTRUCTION,
    TRANSFORM_MESSAGES_INTO_RESEARCH_TOPIC_PROMPT,
)
from deepresearch.scope_level.state import AgentInputState, AgentState
from deepresearch.utils.helper import get_today_str

load_dotenv()

model = init_chat_model(model="openai:gpt-4.1", temperature=0.6)


def clarify_with_user(state: AgentState) -> Command[ScopeNode]:
    """Check the user question contains the sufficient information to proceed"""

    structured_output_model = model.with_structured_output(ClarifyWithUser)
    template = CLARIFY_USER_INSTRUCTION.format(
        messages=get_buffer_string(state[Conversation.messages]),
        date=get_today_str(),
    )

    response: ClarifyWithUser = structured_output_model.invoke(
        [HumanMessage(content=template)]
    )

    if response.need_clarification:
        return Command(
            goto=ScopeNode.END,
            update={Conversation.messages: [AIMessage(content=response.question)]},
        )
    else:
        return Command(
            goto=ScopeNode.WRITE_RESEARCH_BRIEF,
            update={Conversation.messages: [AIMessage(content=response.verification)]},
        )


def write_research_brief(state: AgentState):
    """conversation history into a comprehensive research brief"""

    structured_output_model = model.with_structured_output(ResearchQuestion)
    template = TRANSFORM_MESSAGES_INTO_RESEARCH_TOPIC_PROMPT.format(
        messages=get_buffer_string(state[Conversation.messages]),
        date=get_today_str(),
    )

    response = structured_output_model.invoke(
        [HumanMessage(content=template)]
    )

    state[Conversation.research_brief] = response.research_brief
    state[Conversation.supervisor_messages] = [
        HumanMessage(content=f"{response.research_brief}.")
    ]
    return state


def create_scope_workflow():
    builder = StateGraph(AgentState, input_schema=AgentInputState)

    builder.add_node(ScopeNode.CLARIFY_WITH_USER, clarify_with_user)
    builder.add_node(ScopeNode.WRITE_RESEARCH_BRIEF, write_research_brief)

    builder.add_edge(ScopeNode.START, ScopeNode.CLARIFY_WITH_USER)
    builder.add_edge(ScopeNode.WRITE_RESEARCH_BRIEF, ScopeNode.END)

    return builder
    # scope_research = builder.compile()
