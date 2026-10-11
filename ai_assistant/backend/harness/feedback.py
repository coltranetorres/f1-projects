def submit_feedback(trace_id: str, value: int, reason: str = "", client=None) -> None:
    """Attach a human thumbs score (1 up / 0 down) and free-text reason to a Langfuse trace."""
    if not trace_id:
        raise ValueError("trace_id is required (is Langfuse tracing enabled?)")
    if value not in (0, 1) or isinstance(value, bool):
        raise ValueError("value must be 0 or 1")
    if client is None:
        from langfuse import get_client
        client = get_client()
    client.create_score(trace_id=trace_id, name="user_feedback", value=value,
                        comment=reason.strip() or None)
