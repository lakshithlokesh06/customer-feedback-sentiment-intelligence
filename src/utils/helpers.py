"""Formatting helpers shared by dashboard sections."""


def format_count(value: int) -> str:
    """Format a nonnegative record count for display."""
    return f"{value:,}"
