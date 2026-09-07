"""One Langfuse client and tracing context for API, worker and CLI paths."""

from contextlib import nullcontext
from functools import lru_cache

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from tracemind.config import Settings


@lru_cache(maxsize=4)
def _client(public_key, secret_key, base_url, environment):
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        environment=environment,
        timeout=5,
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
    )


def callbacks_for(settings):
    if client_for(settings) is None:
        return []
    return [CallbackHandler(public_key=settings.langfuse_public_key)]


def observation(client, name, *, as_type="span", **kwargs):
    return (
        client.start_as_current_observation(
            name=name, as_type=as_type, **kwargs
        )
        if client
        else nullcontext()
    )
