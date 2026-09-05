"""Submit an incident or follow-up and watch readable progress over SSE."""

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import httpx

from tracemind.evaluation.models import EvaluationCase
from tracemind.evaluation.evaluator import evaluate_diagnosis
from tracemind.models.result import DiagnosisRun


logger = logging.getLogger("tracemind.demo")
CASE_PATH = (Path(__file__).resolve().parents[1]
             / "evals/cases/payment_failure.json")


def log_result(payload: dict, case: EvaluationCase | None = None) -> None:
    """Separate execution status, actual limits and case acceptance."""
    run = DiagnosisRun.model_validate(payload)
    state = run.state
    if state is not None:
        logger.info(
            "Execution: %s (%s); %.1fs; model calls %d/%d; tool calls %d/%d",
            state.status.value, state.stop_reason,
            (state.elapsed_ms or 0) / 1000,
            state.model_calls, state.limits.model_calls,
            run.tool_call_count, state.limits.tool_calls,
        )
    if case is not None:
        score = evaluate_diagnosis(
            case, run, elapsed_ms=(state.elapsed_ms or 0) if state else 0,
        )
        logger.info("Case %s: %s (case tool budget: %d)", case.id,
                    "passed" if score.passed else "failed scoring",
                    case.max_tool_calls)
    logger.info("诊断结果:\n%s", json.dumps(
        run.diagnosis.model_dump(mode="json"), indent=2, ensure_ascii=False,
    ))


async def watch(client: httpx.AsyncClient, run_id: UUID, *, after: int = 0,
                case: EvaluationCase | None = None):
    """Reconnect with the last delivered sequence; never cancel the worker."""
    cursor = after
    for attempt in range(4):
        try:
            async with client.stream(
                "GET", f"/api/v1/runs/{run_id}/events",
                headers={"Last-Event-ID": str(cursor)},
            ) as response:
                response.raise_for_status()
                event, sequence, data = "", None, []
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("id:"):
                        sequence = int(line[3:].strip())
                    elif line.startswith("data:"):
                        data.append(line[5:].strip())
                    elif not line:
                        if data:
                            payload = json.loads("\n".join(data))
                            if event == "complete":
                                if payload["result_url"]:
                                    result = await client.get(
                                        payload["result_url"]
                                    )
                                    result.raise_for_status()
                                    log_result(result.json(), case)
                                else:
                                    result = await client.get(
                                        f"/api/v1/runs/{run_id}"
                                    )
                                    result.raise_for_status()
                                    status = result.json()
                                    logger.info(
                                        "Execution: %s (%s); model calls %s; "
                                        "tool calls %s",
                                        status["status"],
                                        status["stop_reason"],
                                        status["model_calls"],
                                        status["tool_calls"],
                                    )
                                return payload["status"]
                            if (event == "progress" and sequence is not None
                                    and sequence > cursor):
                                cursor = sequence
                                logger.info("#%s [%s] %s", cursor,
                                            payload["phase"],
                                            payload["message"])
                        event, sequence, data = "", None, []
        except httpx.TransportError as exc:
            logger.warning("订阅连接中断 (%s)", type(exc).__name__)
        if attempt < 3:
            logger.info("从事件 #%s 重新订阅", cursor)
            await asyncio.sleep(.5 * (attempt + 1))
    raise RuntimeError("progress connection closed before completion")


async def run(args) -> str:
    """Use the public API for creation, follow-up and progress subscription."""
    timeout = httpx.Timeout(connect=10, read=None, write=10, pool=10)
    case = None
    async with httpx.AsyncClient(base_url=args.api_url,
                                 timeout=timeout) as client:
        if args.run_id:
            run_id = args.run_id
        elif args.investigation_id:
            body = {"message": args.message, "purpose": args.purpose}
            if args.lookback_minutes is not None:
                end = datetime.now(UTC)
                body.update({
                    "investigation_start": (
                        end - timedelta(minutes=args.lookback_minutes)
                    ).isoformat(),
                    "investigation_end": end.isoformat(),
                })
            response = await client.post(
                f"/api/v1/investigations/{args.investigation_id}/messages",
                json=body,
            )
            response.raise_for_status()
            run_id = UUID(response.json()["id"])
        else:
            case = EvaluationCase.load(args.case)
            end = datetime.now(UTC)
            response = await client.post("/api/v1/investigations", json={
                "incident": case.incident.model_dump(mode="json"),
                "investigation_start": (end - timedelta(
                    minutes=args.lookback_minutes or 10
                )).isoformat(),
                "investigation_end": end.isoformat(),
            })
            response.raise_for_status()
            created = response.json()
            logger.info("Investigation ID: %s", created["id"])
            run_id = UUID(created["run"]["id"])
        logger.info("Run ID: %s", run_id)
        return await watch(client, run_id, after=args.after, case=case)


def main() -> None:
    """Start a real investigation, or subscribe to an existing run."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--investigation-id", type=UUID)
    mode.add_argument("--run-id", type=UUID)
    parser.add_argument("--message")
    parser.add_argument("--purpose", choices=["explain", "investigate"],
                        default="investigate")
    parser.add_argument("--lookback-minutes", type=int)
    parser.add_argument("--after", type=int, default=0)
    parser.add_argument("--case", type=Path, default=CASE_PATH)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.investigation_id and not args.message:
        parser.error("follow-up requires --message")
    if args.message and not args.investigation_id:
        parser.error("--message requires --investigation-id")
    if args.purpose == "explain" and args.lookback_minutes is not None:
        parser.error("explanation uses the previous round's window")
    if args.after < 0 or (args.lookback_minutes is not None
                          and args.lookback_minutes < 1):
        parser.error("invalid cursor or lookback")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        result = asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.info("订阅已关闭，调查仍由后台 worker 执行")
        return
    raise SystemExit(1 if result in {"failed", "cancelled"} else 0)


if __name__ == "__main__":
    main()
