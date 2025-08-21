
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph

from deepresearch.utils.helper import get_today_str
from deepresearch.agent.prompt import FINAL_REPORT_GENERATION_PROMPT
from deepresearch.scope_level.state import AgentInputState, AgentState
from deepresearch.scope_level.agent import clarify_with_user, write_research_brief
from deepresearch.supervisor.multiagent import supervisor_agent

from deepresearch.agent.constants import ConversationConfig, GraphNode


from langchain.chat_models import init_chat_model

writer_model = init_chat_model(model="openai:gpt-4.1")

async def final_report_generation(state: AgentState):
    """Final report"""

    notes = state.get(ConversationConfig.NOTES, [])
    findings = "\n".join(notes)

    final_report_prompt = FINAL_REPORT_GENERATION_PROMPT.format(
        research_brief=state.get(ConversationConfig.RESEARCH_BRIEF, ""),
        findings=findings,
        date=get_today_str()
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])
    return {
        ConversationConfig.FINAL_REPORT: final_report.content,
        ConversationConfig.MESSAGES: ["Here is the final report: " + final_report.content]
    }


deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

deep_researcher_builder.add_node(GraphNode.CLARIFY_WITH_USER, clarify_with_user)
deep_researcher_builder.add_node(GraphNode.WRITE_RESEARCH_BRIEF, write_research_brief)
deep_researcher_builder.add_node(GraphNode.SUPERVISOR_SUBGRAPH, supervisor_agent)
deep_researcher_builder.add_node(GraphNode.FINAL_REPORT_GENERATION, final_report_generation)

deep_researcher_builder.add_edge(GraphNode.START, GraphNode.CLARIFY_WITH_USER)
deep_researcher_builder.add_conditional_edges(
    GraphNode.CLARIFY_WITH_USER,
    lambda state: "end" if state.get("need_clarification", False) else "write_research_brief",
    {
        "end": GraphNode.END,
        "write_research_brief": GraphNode.WRITE_RESEARCH_BRIEF,
    },
)
deep_researcher_builder.add_edge(GraphNode.WRITE_RESEARCH_BRIEF, GraphNode.SUPERVISOR_SUBGRAPH)
deep_researcher_builder.add_edge(GraphNode.SUPERVISOR_SUBGRAPH, GraphNode.FINAL_REPORT_GENERATION)
deep_researcher_builder.add_edge(GraphNode.FINAL_REPORT_GENERATION, GraphNode.END)

agent = deep_researcher_builder.compile()

