"""Project durable run events into readable, reconnectable SSE progress."""

import asyncio
from collections.abc import AsyncIterator
import json
from time import monotonic
from uuid import UUID

from fastapi import Request

from tracemind.agent.storage.run_repository import RunRepository
from tracemind.api.schemas import ProgressEvent
from tracemind.models.run import RunEvent
from tracemind.models.storage import TERMINAL


TOOLS = {
    "query_metrics": "监控指标", "search_traces": "分布式链路",
    "get_trace_detail": "链路详情", "search_logs": "服务日志",
    "read_evidence": "已保存的证据",
    "discover_metrics": "指标名称和标签",
}
MESSAGES = {
    "run.queued": ("queued", "调查已接收，等待执行"),
    "run.started": ("started", "开始调查"),
    "run.resumed": ("resuming", "正在恢复中断的调查"),
    "model.started": ("analyzing", "正在分析已有信息"),
    "model.failed": ("error", "模型调用失败"),
    "model.cancelled": ("cancelling", "模型调用已中止"),
    "recovery.retry_scheduled": ("retrying", "调用暂时失败，正在重试"),
    "model.switched": ("retrying", "正在切换备用模型"),
    "recovery.ambiguous_call": ("resuming", "中断时的调用结果尚未确认"),
    "diagnosis.repair_requested": ("analyzing", "正在修正诊断输出"),
    "policy.finalizing": ("finalizing", "工具调查已停止，正在根据已有证据整理结论"),
    "memory.selected": ("historical_context", "历史经验检索完成，正在核对本次证据"),
    "guidance.selected": ("historical_context", "已读取用户偏好和调查 Skill"),
    "memory.feedback_saved": ("memory_updated", "反馈记忆处理完成"),
}


def readable_event(event: RunEvent) -> ProgressEvent | None:
    """Allowlist display fields; prompts, raw queries and logs stay private."""
    data = {key: value for key, value in event.data.items() if key in {
        "tool", "service", "investigation_start", "investigation_end",
        "result_count", "reason", "status", "error_type", "category",
        "added", "received", "duplicates", "record_count",
        "applied", "skipped", "rejected_count",
    } and isinstance(value, (str, int, float, bool))}
    if event.kind.startswith("tool."):
        tool = TOOLS.get(data.get("tool"), "工具")
        service = data.get("service")
        target = f" {service}" if service else ""
        phase = "querying"
        if event.kind == "tool.started":
            message = f"正在读取{target}的{tool}" if target else f"正在读取{tool}"
        elif event.kind == "tool.completed":
            phase = "query_completed"
            count = data.get("result_count", 0)
            # result_count counts returned preview records, not new evidence.
            message = f"{tool}读取完成，返回 {count} 条记录"
        elif event.kind == "tool.failed":
            phase, message = "error", f"{tool}读取失败"
        elif event.kind == "tool.cancelled":
            phase, message = "cancelling", f"{tool}读取已中止"
        elif event.kind == "tool.skipped":
            phase, message = "finalizing", f"{tool}未执行，调查已进入收尾阶段"
        else:
            return None
    elif event.kind == "observations.stored":
        phase = "evidence_received"
        message = f"已保存 {data.get('added', 0)} 条新增证据"
    elif event.kind == "run.finished":
        phase = "finished"
        message = {
            "succeeded": "诊断已完成",
            "inconclusive": "调查结束，尚未确认根因",
            "cancelled": "调查已取消", "failed": "调查失败",
        }.get(data.get("status"), "调查已结束")
    elif event.kind in MESSAGES:
        phase, message = MESSAGES[event.kind]
    else:
        return None
    return ProgressEvent(
        run_id=event.run_id, sequence=event.sequence,
        span_id=event.span_id, parent_span_id=event.parent_span_id,
        timestamp=event.timestamp, kind=event.kind,
        phase=phase, message=message, data=data,
    )


async def stream_progress(
    repository: RunRepository, run_id: UUID, request: Request, *,
    after: int = 0, poll_seconds: float = .5, heartbeat_seconds: float = 15,
) -> AsyncIterator[str]:
    """Replay progress then tail SQLite; disconnect affects only this reader.

    Durable sequence numbers are cursors. Delivery is at least once across
    reconnects, so clients deduplicate by run ID and sequence. Completion is
    emitted only after the terminal row/result transaction commits.
    """
    cursor = after
    heartbeat = monotonic()
    while not await request.is_disconnected():
        snapshot = await asyncio.to_thread(repository.event_snapshot, run_id)
        for event in snapshot.events:
            if event.sequence <= cursor:
                continue
            cursor = event.sequence
            progress = readable_event(event)
            if progress is not None:
                yield (f"id: {event.sequence}\nevent: progress\n"
                       f"data: {progress.model_dump_json()}\n\n")
        if snapshot.status in TERMINAL:
            result = (f"/api/v1/runs/{run_id}/result"
                      if snapshot.has_result else None)
            yield (f"id: {cursor}\nevent: complete\ndata: " + json.dumps({
                "run_id": str(run_id), "status": snapshot.status.value,
                "result_url": result,
            }) + "\n\n")
            return
        if monotonic() - heartbeat >= heartbeat_seconds:
            yield ": keep-alive\n\n"
            heartbeat = monotonic()
        await asyncio.sleep(poll_seconds)
