"""
Unit tests for each node in isolation.
We mock the LLM and the HTTP client so tests run without credentials.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.agent.nodes import intent_node, fetch_node, synthesize_node


# intent_node, happy path
@pytest.mark.asyncio
async def test_intent_node_happy_path():
    """Should extract country and fields from a clear question."""
    mock_response = MagicMock()
    mock_response.content = '{"country_name": "Germany", "requested_fields": ["population"]}'

    with patch("app.agent.nodes._get_llm") as mock_llm_fn:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm

        result = await intent_node({"user_query": "What is the population of Germany?"})

    assert result["country_name"] == "Germany"
    assert "population" in result["requested_fields"]
    assert result["intent_error"] is None


# intent_node, no country
@pytest.mark.asyncio
async def test_intent_node_no_country():
    """Should set intent_error when no country is found in the question."""
    mock_response = MagicMock()
    mock_response.content = '{"country_name": null, "requested_fields": []}'

    with patch("app.agent.nodes._get_llm") as mock_llm_fn:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm

        result = await intent_node({"user_query": "What is the weather like?"})

    assert result["country_name"] is None
    assert result["intent_error"] is not None


# fetch_node, skip on intent error
@pytest.mark.asyncio
async def test_fetch_node_skips_on_intent_error():
    """Should skip the API call when intent_error is already set."""
    state = {
        "intent_error": "I couldn't identify a country.",
        "country_name": None,
    }
    result = await fetch_node(state)
    assert result["raw_country_data"] is None
    assert result["fetch_error"] is None


# fetch_node, country not found
@pytest.mark.asyncio
async def test_fetch_node_not_found():
    """Should set fetch_error when the country doesn't exist."""
    from app.tools.countries_api import CountryNotFoundError

    with patch("app.agent.nodes.fetch_country",
               AsyncMock(side_effect=CountryNotFoundError("Not found"))):
        result = await fetch_node({
            "intent_error": None,
            "country_name": "Narnia",
        })

    assert result["raw_country_data"] is None
    assert "Narnia" in result["fetch_error"] or result["fetch_error"]


# synthesize_node, intent error
@pytest.mark.asyncio
async def test_synthesize_node_passes_through_intent_error():
    """Should return intent_error as the answer without calling the LLM."""
    state = {
        "intent_error": "I couldn't identify a country.",
        "fetch_error": None,
        "raw_country_data": None,
        "user_query": "blah",
        "requested_fields": [],
    }
    result = await synthesize_node(state)
    assert result["answer"] == "I couldn't identify a country."


# synthesize_node, happy path
@pytest.mark.asyncio
async def test_synthesize_node_happy_path():
    """Should call the LLM and return its answer."""
    mock_response = MagicMock()
    mock_response.content = "Germany has a population of approximately 83,240,000."

    state = {
        "intent_error": None,
        "fetch_error": None,
        "user_query": "What is the population of Germany?",
        "requested_fields": ["population"],
        "raw_country_data": [{"name": {"common": "Germany"}, "population": 83240000}],
    }

    with patch("app.agent.nodes._get_llm") as mock_llm_fn:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_response
        mock_llm_fn.return_value = mock_llm

        result = await synthesize_node(state)

    assert "83,240,000" in result["answer"] or "Germany" in result["answer"]