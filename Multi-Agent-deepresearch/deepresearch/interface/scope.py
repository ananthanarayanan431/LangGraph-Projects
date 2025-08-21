from typing import Any, Dict, List, Optional
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from deepresearch.scope_level.agent import create_scope_workflow
from deepresearch.scope_level.state import AgentInputState

SCOPE_LEVEL_TAG = "Scope Level"

router = APIRouter(prefix="/scope", tags=[SCOPE_LEVEL_TAG])
