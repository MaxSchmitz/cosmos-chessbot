"""Utility functions and helpers."""

import json
import logging
from typing import Optional


def extract_json(text: str) -> Optional[dict]:
    """Extract the first valid JSON object from text by tracking brace depth.

    Unlike a simple regex like ``r'{[^}]+}'``, this correctly handles
    nested braces (arrays, nested objects) that appear in Cosmos responses.

    Args:
        text: Raw text potentially containing a JSON object

    Returns:
        Parsed dict, or None if no valid JSON object found
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]

        if escape:
            escape = False
            continue

        if c == "\\":
            if in_string:
                escape = True
            continue

        if c == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Try next opening brace
                    next_start = text.find("{", start + 1)
                    if next_start == -1:
                        return None
                    return extract_json(text[next_start:])

    return None


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging for the application.

    Args:
        verbose: Enable DEBUG level logging
        quiet: Suppress INFO, show only WARNING+
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
