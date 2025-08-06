
from pprint import pprint
from dotenv import load_dotenv

from graph.chains.model import RouterQuery

load_dotenv()

from graph.chains.generation import generation_cahin
from graph.chains.hallucination_grader import GradeHallucination, hallucination_grader
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.answer_grader import GradeAnswer, answer_grader
from graph.chains.router import question_router

from ingestion import retriever


def test_retrieval_grader_answer_yes():
    """Test that the retrieval grader returns yes for a relevant document"""
    question = "what is agent"
    docs = retriever.invoke(question)
    doc_text = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke({"question": question, "document": doc_text})
    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no():
    """Test that the retrieval grader returns no for a irrelevant document"""
    question = "how to train your dragon"
    docs = retriever.invoke(question)
    doc_text = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke({"question": question, "document": doc_text})
    assert res.binary_score == "no"


def test_generation_chain():
    question = "what is agent"
    docs = retriever.invoke(question)
    generation = generation_cahin.invoke({"question": question, "context": docs})
    print(generation)


def test_hallucination_grader_answer_yes():
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_cahin.invoke({"question": question, "context": docs})
    res: GradeHallucination = hallucination_grader.invoke({"question": question, "answer": generation})
    assert res.binary_score == "yes"


def test_hallucination_grader_answer_no():
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_cahin.invoke({"question": question, "context": docs})
    res: GradeHallucination = hallucination_grader.invoke({"question": question, "answer": generation})
    assert res.binary_score == "no"


def test_router_to_vectorstore():
    question = "agent memory"
    res : RouterQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"

def test_router_to_websearch():
    question = "How to train your dragon"
    res : RouterQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"

