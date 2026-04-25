"""
FastAPI route definitions.

POST /ask   — The main endpoint: takes a question, returns an answer.
GET  /health — Alive check (used by hosting platforms).
"""

from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.agent.graph import get_graph

logger = logging.getLogger(__name__)
router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["What is the population of Germany?"],
    )


class AskResponse(BaseModel):
    answer: str
    country: str | None = None


@router.post("/ask", response_model = AskResponse)
async def ask(body: AskRequest) -> AskResponse:
    """
    Ask a question about a country.

    The agent will:
    1. Identify the country and fields from your question.
    2. Fetch live data from the REST Countries API.
    3. Return a natural-language answer.
    """
    logger.info("Received question: %r", body.question)

    graph = get_graph()

    # Build the initial state, only user_query is required to start
    initial_state = {
        "user_query": body.question,
        "country_name": None,
        "requested_fields": [],
        "intent_error": None,
        "raw_country_data": None,
        "fetch_error": None,
        "answer": "",
    }

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("Unhandled error in graph execution")
        raise HTTPException(status_code=500, detail="Internal agent error.") from exc

    return AskResponse(
        answer=final_state["answer"],
        country=final_state.get("country_name"),
    )


@router.get("/health")
async def health():
    """Alive check - returns 200 if the service is running."""
    return {"status": "ok"}