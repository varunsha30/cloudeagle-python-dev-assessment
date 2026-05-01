"""
The three nodes of the Country Agent graph, using an LLM.

Node 1 — intent_node: LLM parses the user's question
Node 2 — fetch_node: HTTP call to REST Countries API (no LLM)
Node 3 — synthesize_node: LLM writes the final answer

Each node receives the full AgentState and returns a partial dict.
LangGraph merges it into the state automatically.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Literal
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from app.agent.state import AgentState
from app.config import get_settings
from app.tools.countries_api import (
    fetch_country,
    CountryNotFoundError,
    CountriesAPIError,
)

logger = logging.getLogger(__name__)

# Field map
# Maps intent-parser field names to actual REST Countries v3.1 JSON keys.
# Add new entries here without touching any other file.
FIELD_MAP: dict[str, str] = {
    "name":          "name",
    "capital":       "capital",       # list: ["Berlin"]
    "region":        "region",
    "subregion":     "subregion",
    "continent":     "continents",    # list: ["Europe"]
    "borders":       "borders",       # list of ISO-3 codes
    "population":    "population",    # integer
    "area":          "area",          # float, km²
    "languages":     "languages",     # dict: {"deu": "German"}
    "demonym":       "demonyms",
    "currency":      "currencies",    # dict: {"EUR": {"name": "Euro", "symbol": "€"}}
    "currencies":    "currencies",
    "calling_code":  "idd",           # dict: {"root": "+4", "suffixes": ["9"]}
    "tld":           "tld",           # list: [".de"]
    "latlng":        "latlng",
    "landlocked":    "landlocked",
    "timezone":      "timezones",     # list: ["UTC+01:00"]
    "timezones":     "timezones",
    "flag":          "flag",          # emoji string: "🇩🇪"
    "flag_image":    "flags",         # dict: {"png": "...", "svg": "...", "alt": "..."}
    "independent":   "independent",
    "un_member":     "unMember",
    "translations":  "translations",
    "maps":          "maps",
    "car":           "car",
    "fifa":          "fifa",
    "gini":          "gini",
}

ALL_FIELDS = sorted(FIELD_MAP.keys())


def _get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    Return a ChatGoogleGenerativeAI instance.
    temperature=0 for deterministic JSON extraction (intent node).
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )

class IntentResult(BaseModel):
    """Structured output schema for the intent parsing node."""
    country_name: str | None = Field(
        default=None,
        description="The country name to look up, e.g. 'Germany'. Null if not found."
    )
    requested_fields: list[str] = Field(
        default_factory=list,
        description="List of country data fields the user wants, from the supported list."
    )


# Node 1: Intent or Field Identification

INTENT_SYSTEM_PROMPT = """\
You are an intent parser for a country-information assistant.

Given a user question:
1. Extract the country name as a standard search term (e.g. "Germany" not "German").
2. Extract the fields the user wants from this supported list: {fields}

Rules:
- Normalise country names: "UK" → "United Kingdom", "USA" → "United States"
- If no specific fields are mentioned, infer the most relevant ones from context.
- If the question is not about a country at all, set country_name to null.
- Only use field names from the supported list above.
"""

async def intent_node(state: AgentState) -> Command[Literal["fetch_node", "synthesize_node"]]:
    """
    Node 1: Intent or Field Identification

    Uses the LLM (JSON mode) to extract the country name
    and the fields the user is asking about.

    Returns partial state with: country_name, requested_fields, intent_error
    """
    query = state["user_query"]
    logger.info("[intent_node] Parsing: %r", query)

    llm = _get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(IntentResult)
    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT.format(
            fields=", ".join(ALL_FIELDS)
        )),
        HumanMessage(content=query),
    ]

    try:
        result: IntentResult = await structured_llm.ainvoke(messages)
        country_name = result.country_name
        requested_fields = result.requested_fields

        valid_fields = [f for f in requested_fields if f in FIELD_MAP]

        if not country_name:
            return Command(
                update={
                    "country_name": None,
                    "requested_fields": [],
                    "intent_error": (
                        "I couldn't identify a country in your question. "
                        "Try asking: 'What is the population of France?'"
                    ),
                },
                goto="synthesize_node"
            )

        logger.info("[intent_node] country=%r fields=%r", country_name, valid_fields)
        return Command(
            update={
                "country_name": country_name,
                "requested_fields": valid_fields or ["population", "capital"],
                "intent_error": None,
            },
            goto="fetch_node"
        )

    except Exception as exc:
        logger.error("[intent_node] Parse failed: %s", exc)
        return Command(
            update={
                "country_name": None,
                "requested_fields": [],
                "intent_error": (
                    "I had trouble understanding your question. Could you rephrase it?"
                ),
            },
            goto="synthesize_node"
        )


# Node 2: Tool Invocation (API fetch)

async def fetch_node(state: AgentState) -> Command[Literal["synthesize_node"]]:
    """
    Node 2: Tool Invocation

    Calls the REST Countries API. No LLM involved, pure HTTP.
    Skips gracefully if intent_node already set an error.
    """
    if state.get("intent_error"):
        return Command(
            update={"raw_country_data": None, "fetch_error": None},
            goto="synthesize_node"
        )

    country_name = state["country_name"]
    logger.info("[fetch_node] Fetching: %r", country_name)

    try:
        data = await fetch_country(country_name)
        logger.info("[fetch_node] Retrieved %d match(es)", len(data))
        return Command(
            update={"raw_country_data": data, "fetch_error": None},
            goto="synthesize_node"
        )

    except CountryNotFoundError as exc:
        return Command(
            update={"raw_country_data": None, "fetch_error": str(exc)},
            goto="synthesize_node"
        )

    except CountriesAPIError as exc:
        logger.error("[fetch_node] API error: %s", exc)
        return Command(
            update={
                "raw_country_data": None,
                "fetch_error": (
                    "I'm having trouble reaching the countries database. "
                    "Please try again in a moment."
                ),
            },
            goto="synthesize_node"
        )


# Node 3: Answer Synthesis
# The synthesis prompt explicitly describes the nested JSON shapes
# that appear in the REST Countries API.

SYNTHESIZE_SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about countries.

You will receive:
1. The user's original question.
2. A list of fields they asked about.
3. Structured JSON data from a countries database.

Important — the data uses these shapes:
- capital: a list, e.g. ["Berlin"]  → read as "the capital is Berlin"
- currencies: a dict keyed by code, e.g. {"EUR": {"name": "Euro", "symbol": "€"}}
- languages: a dict keyed by code, e.g. {"deu": "German"}
- timezones: a list, e.g. ["UTC+01:00", "UTC+02:00"]
- continents: a list, e.g. ["Europe"]
- borders: a list of 3-letter ISO codes, e.g. ["AUT", "BEL"]

Rules:
- Answer in 1–3 conversational sentences.
- Answer ONLY from the provided data — never invent facts.
- Format population numbers with commas (83,240,000 not 83240000).
- For currencies, include both the name and symbol.
- If a requested field is missing from the data, say so explicitly.
- Do not mention "the database", "the API", or "the JSON".
"""


def _extract_fields(country_data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Pull only the requested fields out of a raw country object."""
    extracted: dict[str, Any] = {
        "name": country_data.get("name", {}).get("common", "Unknown")
    }
    for field in fields:
        api_key = FIELD_MAP.get(field, field)
        if api_key in country_data:
            extracted[field] = country_data[api_key]
    return extracted


async def synthesize_node(state: AgentState) -> dict:
    """
    Node 3: Answer Synthesis

    Composes a natural-language answer from the raw API data.

    Error waterfall (no LLM call needed for errors):
      intent_error -> return that message directly
      fetch_error -> return that message directly
      no data -> return a graceful fallback
      happy path -> ask the LLM to synthesise
    """
    # Error waterfall
    if state.get("intent_error"):
        logger.info("[intent_node] Intent Erorr, incorrect intent: %s", state["intent_error"])
        return {"answer": state["intent_error"]}
    if state.get("fetch_error"):
        logger.info("[fetch_node] Fetch Erorr, could not fetch from API: %s", state["fetch_error"])
        return {"answer": state["fetch_error"]}

    raw_data = state.get("raw_country_data")
    if not raw_data:
        return {"answer": "I couldn't find any information for that country."}

    country = raw_data[0]
    fields = state.get("requested_fields") or ["population", "capital"]
    extracted = _extract_fields(country, fields)

    logger.info(
        "[synthesize_node] Synthesising for %r with fields %r",
        extracted.get("name"), fields
    )

    # For synthesis a tiny bit of temperature so answers sound natural
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.2
    )

    user_message = (
        f"User question: {state['user_query']}\n\n"
        f"Fields requested: {fields}\n\n"
        f"Country data: {json.dumps(extracted, indent=2, ensure_ascii=False)}"
    )
    messages = [
        SystemMessage(content=SYNTHESIZE_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    try:
        response = await llm.ainvoke(messages)
        answer = response.content.strip()
        logger.info("[synthesize_node] Found Answer: %s", answer)
        return {"answer": answer}

    except Exception as exc:
        logger.error("[synthesize_node] LLM failed: %s", exc)
        return {
            "answer": (
                f"I found the data but had trouble composing an answer. "
                f"Raw facts: {json.dumps(extracted, indent=2, ensure_ascii=False)}"
            )
        }