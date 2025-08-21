
import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    filter_messages
)

from langgraph.graph import StateGraph
from langgraph.types import Command

from deepresearch.supervisor.prompt import LEAD_RESEARCHER_PROMPT
from deepresearch.research_level.agent import research_agent
from deepresearch.supervisor.state import SupervisorState
from deepresearch.supervisor.tools import (
    ConductResearch,
    ResearchComplete,
    think_tool
)

from deepresearch.supervisor.constants import ToolName, ConversationConfig, GraphNode
from deepresearch.utils.helper import get_today_str


def get_notes_from_tool_calls(messages: list[BaseMessage])->list[str]:
    """
    This function retrieves the compressed research findings that sub-agents
    return as ToolMessage content. When the supervisor delegates research to
    sub-agents via ConductResearch tool calls, each sub-agent returns its
    compressed findings as the content of a ToolMessage. This function
    extracts all such ToolMessage content to compile the final research notes.
    """

    return [
        tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")
    ]

supervisor_tool = [think_tool, ConductResearch, ResearchComplete]
supervisor_model = init_chat_model(model="openai:gpt-4.1")
supervisor_model_with_tools = supervisor_model.bind_tools(tools=supervisor_tool)

max_researcher_iteration = 6
max_concurrent_researcher = 3

async def supervisor(state: SupervisorState) -> Command[Literal[ToolName.SUPERVISOR_TOOLS]]:

    supervisor_messages = state.get(ConversationConfig.supervisor_messages, [])
    system_messages = LEAD_RESEARCHER_PROMPT.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researcher,
        max_researcher_iterations=max_researcher_iteration
    )
    messages = [SystemMessage(content=system_messages)] + supervisor_messages
    response = await supervisor_model_with_tools.ainvoke(messages)

    return Command(
        goto=ToolName.SUPERVISOR_TOOLS,
        update={
            ConversationConfig.supervisor_messages: [response],
            ConversationConfig.research_iterations: state.get(ConversationConfig.research_iterations, 0) + 1 
        }
    )
    
async def supervisor_tools(state: SupervisorState)-> Command[Literal[ToolName.SUPERVISOR, GraphNode.END]]:
    
    supervisor_messages = state.get(ConversationConfig.supervisor_messages, [])
    research_iteration = state.get(ConversationConfig.research_iterations, 0)
    most_recent_message = supervisor_messages[-1]

    tool_messages = []
    all_raw_notes = []
    next_step = ToolName.SUPERVISOR
    should_end = False

    exceed_iteration = research_iteration >= max_researcher_iteration
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call['name']==ToolName.RESEARCH_COMPLETE
        for tool_call in most_recent_message.tool_calls
    )

    if exceed_iteration or no_tool_calls or research_complete:
        should_end = True
        next_step = GraphNode.END
    
    else:
        try:
            think_tools_call = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call['name'] == ToolName.THINK_TOOL
            ]

            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call['name'] == ToolName.CONDUCT_RESEARCH
            ]

            for tool_call in think_tools_call:
                observation = think_tool.invoke(tool_call['args'])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )
            
            if conduct_research_calls:
                coros = [
                    research_agent.ainvoke({
                        ConversationConfig.supervisor_messages : [HumanMessage(content=tool_call["args"]["research_topic"])],
                        ConversationConfig.research_topic: tool_call["args"]["research_topic"]
                    }) 
                    for tool_call in conduct_research_calls
                ]
                tool_results = await asyncio.gather(*coros)

                research_tool_messages = [
                    ToolMessage(
                        content=result.get(GraphNode.COMPRESSED_RESEARCH, "Error synthesizing research report"),
                        name = tool_call['name'],
                        tool_call_id = tool_call['id']
                    ) for result, tool_call in zip(tool_results, conduct_research_calls)
                ]

                tool_messages.extend(research_tool_messages)

                all_raw_notes = [
                    "\n".join(result.get("raw_notes", [])) 
                    for result in tool_results
                ]
            
        except Exception as e:
            should_end = True
            next_step = END

        
    if should_end:
        return Command(
            goto=next_step,
            update={
                ConversationConfig.notes: get_notes_from_tool_calls(supervisor_messages),
                ConversationConfig.research_brief: state.get(ConversationConfig.research_brief, "")
            }
        )
    else:
        return Command(
            goto=next_step,
            update={
                ConversationConfig.supervisor_messages: tool_messages,
                ConversationConfig.raw_notes: all_raw_notes
            }
        )


supervisor_builder = StateGraph(SupervisorState)

supervisor_builder.add_node(ToolName.SUPERVISOR, supervisor)
supervisor_builder.add_node(ToolName.SUPERVISOR_TOOLS, supervisor_tools)

supervisor_builder.add_edge(GraphNode.START, ToolName.SUPERVISOR)
supervisor_agent = supervisor_builder.compile()

