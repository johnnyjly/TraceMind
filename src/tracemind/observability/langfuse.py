"""One Langfuse client and tracing context for API, worker and CLI paths."""

from contextlib import nullcontext
from functools import lru_cache

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from tracemind.config import Settings


@lru_cache(maxsize=4)
def _client(public_key, secret_key, base_url, environment, timeout):
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        environment=environment,
        timeout=timeout,
        additional_headers={"x-langfuse-ingestion-version": "4"},
    )


def client_for(settings: Settings, *, required=False):
    """Do not construct an exporter until explicitly enabled and configured."""
    if not settings.langfuse_enabled:
        if required:
            raise ValueError(
                "TRACEMIND_LANGFUSE_ENABLED must be true for experiments"
            )
        return None
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        raise ValueError(
            "Langfuse project public and secret keys are required"
        )
    return _client(
        settings.langfuse_public_key,
        settings.langfuse_secret_key.get_secret_value(),
        str(settings.langfuse_base_url).rstrip("/"),
        settings.environment,
        settings.langfuse_timeout_seconds,
    )


@lru_cache(maxsize=4)
def _callback(public_key):
    # Nested graph/tools/models must inherit the same handler identity so
    # LangChain deduplicates explicit and inherited callbacks per call.
    return CallbackHandler(public_key=public_key)


def callbacks_for(settings):
    if client_for(settings) is None:
        return []
    return [_callback(settings.langfuse_public_key)]


def observation(client, name, *, as_type="span", **kwargs):
    return (
        client.start_as_current_observation(
            name=name, as_type=as_type, **kwargs
        )
        if client
        else nullcontext()
    )
