
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from graph.chains.generation import generation_cahin
from graph.chains.retrieval_grader import retrieval_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from graph.chains.router import question_router, RouterQuery

from graph.constants import GraphNode
from graph.nodes.generate import generate
from graph.nodes.grade_documents import grade_documents
from graph.nodes.retrieve import retrieve
from graph.nodes.web_search import web_search

from graph.state import AgentState

load_dotenv()

def decide_to_generate(state: AgentState):
    if state["web_search"]:
        return GraphNode.WEBSEARCH
    else:
        return GraphNode.GENERATE

def grade_generation_grounded_in_documents_and_question(state: AgentState):

    question = state["question"]
    documents = state['documents']
    generation = state['generation']

    score = hallucination_grader.invoke({"document": documents, "generation": generation})
    if hallucination_grade := score.binary_score:

        score = answer_grader.invoke({"document": documents, "generation": generation})
        if answer_grade := score.binary_score:
            return GraphNode.USEFUL
        else:
            return GraphNode.NO_USEFUL
    else:
        return GraphNode.NOT_SUPPORTED
    

def route_question(state: AgentState):
    question = state["question"]
    source: RouterQuery = question_router.invoke({"question": question})
    if source.datasource == "vectorstore":
        return GraphNode.RETRIEVE
    else:
        return GraphNode.WEBSEARCH
    
    
builder = StateGraph(AgentState)
builder.add_node(GraphNode.RETRIEVE, retrieve)
builder.add_node(GraphNode.GRADE_DOCUMENTS, grade_documents)
builder.add_node(GraphNode.GENERATE, generate)
builder.add_node(GraphNode.WEBSEARCH, web_search)

builder.set_conditional_entry_point(
    route_question,
    {
        GraphNode.RETRIEVE: GraphNode.RETRIEVE,
        GraphNode.WEBSEARCH: GraphNode.WEBSEARCH,
    },
)
builder.add_edge(GraphNode.RETRIEVE, GraphNode.GRADE_DOCUMENTS)
builder.add_conditional_edges(
    GraphNode.GRADE_DOCUMENTS,
    decide_to_generate,
    {
        GraphNode.WEBSEARCH: GraphNode.WEBSEARCH,
        GraphNode.GENERATE: GraphNode.GENERATE,
    },
)

builder.add_conditional_edges(
    GraphNode.GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        GraphNode.USEFUL: END,
        GraphNode.NO_USEFUL: GraphNode.WEBSEARCH,
        GraphNode.NOT_SUPPORTED: GraphNode.GENERATE,
    },
)

builder.add_edge(GraphNode.WEBSEARCH, GraphNode.GENERATE)
builder.add_edge(GraphNode.GENERATE, END)

app = builder.compile()


app.get_graph().draw_mermaid_png(output_file_path="graph.png")