
from typing import Literal, List
from pydantic import BaseModel, Field


class RouterQuery(BaseModel):
    """Route a user query to the most relevant datasource"""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user query, determine which datasource is most relevant to answer the query"
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""

    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GradeHallucination(BaseModel):
    """Binary score for hallucination present in the generation"""

    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Answer is grounded in the documents, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score for the answer"""

    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Answer addresses the user query, 'yes' or 'no'"
    )

