"""
The AgentState TypedDict is the single source of truth that all
LangGraph nodes read from and write to.
"""

from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Input
    user_query: str

    # Set by: intent_node
    country_name: Optional[str]
    requested_fields: list[str]
    intent_error: Optional[str]

    # Set by: fetch_node
    raw_country_data: Optional[list[dict[str, Any]]]
    fetch_error: Optional[str]

    # Set by: synthesize_node
    answer: str