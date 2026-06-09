from langfuse import Langfuse
import config

_client: Langfuse | None = None


def get_langfuse() -> Langfuse | None:
    """Returns the Langfuse client, or None if keys are not configured."""
    global _client
    if not config.LANGFUSE_PUBLIC_KEY or not config.LANGFUSE_SECRET_KEY:
        return None
    if _client is None:
        _client = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )
    return _client


def flush():
    if _client:
        _client.flush()
