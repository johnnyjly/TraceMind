"""Count paid keywords in the Harness; retrieve over HTTP."""

import httpx

from tracemind.rag.config import RagSettings
from tracemind.rag.errors import RetrievalUnavailable
from tracemind.rag.model_bridge import Completion, LightRAGModelBridge
from tracemind.rag.query_contracts import (
    PreparedQuery,
    QueryKeywords,
    RetrievalRequest,
    RetrievalResponse,
)


class HttpSnapshotRetriever:
    def __init__(self, settings: RagSettings, *, langfuse_client=None):
        self.settings = settings
        self.langfuse = langfuse_client

    async def __call__(
        self,
        queries: list[str],
        *,
        question: str,
        snapshot: str,
        completion: Completion,
    ):
        # Keep the pinned LightRAG keyword prompt while leaving credentials,
        # admission and usage accounting in the investigation process.
        from lightrag.constants import DEFAULT_SUMMARY_LANGUAGE
        from lightrag.prompt import PROMPTS

        bridge = LightRAGModelBridge(self.settings, completion)
        prepared = []
        async with httpx.AsyncClient(
            base_url=str(self.settings.service_url),
            timeout=self.settings.timeout_seconds,
            trust_env=False,
        ) as client:
            ready = await client.get("/ready")
            ready.raise_for_status()
            if ready.json().get("snapshot") != snapshot:
                raise RetrievalUnavailable("knowledge snapshot mismatch")
            for query in queries:
                if self.settings.mode == "naive":
                    keywords = QueryKeywords(low_level_keywords=[query])
                else:
                    prompt = PROMPTS["keywords_extraction"].format(
                        query=query,
                        examples="\n".join(
                            PROMPTS["keywords_extraction_examples"]
                        ),
                        language=DEFAULT_SUMMARY_LANGUAGE,
                    )
                    text = await bridge.complete(
                        prompt, keyword_extraction=True
                    )
                    keywords = QueryKeywords.model_validate_json(text)
                    if (
                        not keywords.high_level_keywords
                        and not keywords.low_level_keywords
                    ):
                        raise RetrievalUnavailable("no retrieval keywords")
                prepared.append(
                    PreparedQuery(query=query, **keywords.model_dump())
                )
            request = RetrievalRequest(
                snapshot=snapshot,
                mode=self.settings.mode,
                question=question,
                queries=prepared,
            )
            headers = {}
            if self.langfuse is not None:
                trace_id = self.langfuse.get_current_trace_id()
                parent_id = self.langfuse.get_current_observation_id()
                if trace_id and parent_id:
                    headers["traceparent"] = f"00-{trace_id}-{parent_id}-01"
            response = await client.post(
                "/search", json=request.model_dump(), headers=headers
            )
            response.raise_for_status()
            result = RetrievalResponse.model_validate(response.json())
            if result.snapshot != snapshot:
                raise RetrievalUnavailable("knowledge snapshot mismatch")
            return result
