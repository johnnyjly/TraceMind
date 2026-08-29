"""Public agent interfaces, loaded only when requested by a caller."""

from importlib import import_module


_EXPORTS = {
    "DiagnosisError": "tracemind.agent.errors",
    "DiagnosisRun": "tracemind.models.result",
    "DiagnosisService": "tracemind.agent.service",
    "LangChainAgentRunner": "tracemind.agent.runtime.runner",
    "ModelConfigurationError": "tracemind.agent.errors",
    "create_chat_model": "tracemind.agent.runtime.llm",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    # Preserve the public facade without importing model SDKs for storage use.
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
