"""Formatting utilities for environment loaders."""


def wrap_between_tags(text: str, tag_name: str) -> str:
    """
    Wrap text in XML-style tags.

    Args:
        text: Content to wrap
        tag_name: Tag name (e.g., "user", "agent", "tool_call")

    Returns:
        Formatted string like "<tag_name>\\ntext\\n</tag_name>"
    """
    return f"<{tag_name}>\n{text}\n</{tag_name}>"
