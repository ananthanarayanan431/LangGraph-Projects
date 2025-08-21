
from email import message
from typing import Literal
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, filter_messages
from langchain.chat_models import init_chat_model

from deepresearch.research_level.state import ResearcherOutputState, ResearcherState
from deepresearch.research_level.tools import tavily_search, think_tool
from deepresearch.utils.helper import get_today_str
from deepresearch.research_level.prompt import (
    RESEARCH_AGENT_PROMPT,
    COMPRESS_RESEARCH_SYSTEM_PROMPT,
    COMPRESS_RESEARCH_HUMAN_MESSAGE
)

from deepresearch.research_level.constant import ResearchNode, ConversationResearch

tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}


model = init_chat_model(model="openai:gpt-4.1")
model_with_tools = model.bind_tools(tools)

summarization_model = init_chat_model(model="openai:gpt-4.1-mini")
compress_model = init_chat_model(model="openai:gpt-4.1")


def llm_call(state: ResearcherState):
    """Analyze the current state and determine the next step"""

    response = model_with_tools.invoke(
        [SystemMessage(content=RESEARCH_AGENT_PROMPT)] + state[ConversationResearch.researcher_messages]
    )
    return {
        ConversationResearch.researcher_messages: [response]
    }

def tool_node(state: ResearcherState):

    tool_calls = state[ConversationResearch.researcher_messages][-1].tool_calls

    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))
    
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]
    state[ConversationResearch.researcher_messages] = tool_outputs
    return state

def compress_research(state: ResearcherState):

    system_message = COMPRESS_RESEARCH_SYSTEM_PROMPT.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get(ConversationResearch.researcher_messages, []) + [HumanMessage(content=COMPRESS_RESEARCH_HUMAN_MESSAGE)]

    response = compress_model.invoke(messages)

    raw_notes = [
        str(m.content) for m in filter_messages(
            state[ConversationResearch.researcher_messages], 
            include_types=["tool", "ai"]
        )
    ]
    state[ConversationResearch.compressed_research] = str(response.content)
    state[ConversationResearch.raw_notes] = ["\n".join(raw_notes)]
    return state 


def should_continue(state: ResearcherState)->Literal[ResearchNode]:

    messages = state[ConversationResearch.researcher_messages]
    last_messages = messages[-1]

    if last_messages.tool_calls:
        return ResearchNode.TOOL_NODE
    
    return ResearchNode.COMPRESS_RESEARCH


def create_research_workflow():
    builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

    builder.add_node(ResearchNode.LLM_CALL, llm_call)
    builder.add_node(ResearchNode.TOOL_NODE, tool_node)
    builder.add_node(ResearchNode.COMPRESS_RESEARCH, compress_research)

    builder.add_edge(ResearchNode.START, ResearchNode.LLM_CALL)
    builder.add_conditional_edges(
    ResearchNode.LLM_CALL,
    should_continue,
    {
        ResearchNode.TOOL_NODE: ResearchNode.TOOL_NODE,
        ResearchNode.COMPRESS_RESEARCH: ResearchNode.COMPRESS_RESEARCH,
    },
    )

    builder.add_edge(ResearchNode.TOOL_NODE, ResearchNode.LLM_CALL)
    builder.add_edge(ResearchNode.COMPRESS_RESEARCH, ResearchNode.END)

    return builder
    

research_agent = create_research_workflow().compile()