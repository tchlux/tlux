"""Query planner converting JSON into ``QuerySpec`` objects."""

import json
from ..schema import QuerySpec


def parse_query(json_str: str) -> QuerySpec:
    """Parse a JSON string into a :class:`QuerySpec`."""
    obj = json.loads(json_str)
    return QuerySpec(
        embeddings=obj.get("embeddings", []),
        token_sequence=obj.get("token_sequence", []),
        label_include=obj.get("label_include", {}),
        numeric_range=obj.get("numeric_range", {}),
        top_k=obj.get("top_k", 10),
    )
