"""Capture one controlled fault window with correlated requests
and preflight.
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import signal
import time
from uuid import uuid4

import httpx

from tracemind.evaluation.demo import (
    SCENARIO_NAMES,
    compose_command,
    docker_environment,
    prepare_flag_config,
    verify_demo_commit,
)
from tracemind.evaluation.models import EvaluationCase
from tracemind.mcp.client import HttpMcpClient

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def request(method, path, data=None):
    trace_id = uuid4().hex
    parent_id = uuid4().hex[:16]
    row = {
        "method": method,
        "path": path,
        "trace_id": trace_id,
        "started_at": datetime.now(UTC).isoformat(),
    }
    started = time.monotonic()
    try:
        with httpx.Client(timeout=40, trust_env=False) as client:
            response = client.request(
                method,
                "http://127.0.0.1:8080" + path,
                json=data,
                headers={"traceparent": f"00-{trace_id}-{parent_id}-01"},
            )
        row["status"] = response.status_code
        if response.is_error:
            row["error_body"] = response.text[:2000]
        elif method == "GET" and path.startswith("/api/cart"):
            row["cart_items"] = len(response.json().get("items", []))
    except httpx.HTTPError as exc:
        row["error_type"] = type(exc).__name__
    row["seconds"] = time.monotonic() - started
    return row


def traffic(case, people):
    us = next(p for p in people if p["address"]["country"] == "United States")
    foreign = next(
        p for p in people if p["address"]["country"] != "United States"
    )

    def order(person):
        user = "ops-rca-" + uuid4().hex
        identity = sha256(user.encode()).hexdigest()
        rows = [
            request(
                "POST",
                "/api/cart",
                {
                    "userId": user,
                    "item": {"productId": "0PUK6V6EV0", "quantity": 1},
                },
            )
        ]
        rows.append(
            {
                **request("POST", "/api/checkout", {**person, "userId": user}),
                "country": person["address"]["country"],
            }
        )
        if case == "cart_failure":
            rows.append(request("DELETE", "/api/cart", {"userId": user}))
            rows.append(request("GET", "/api/cart?sessionId=" + user))
        for row in rows:
            row["user_hash"] = identity
        return rows

    with ThreadPoolExecutor(max_workers=3) as pool:
        if case == "product_catalog_lock_contention":
            tasks = [
                pool.submit(request, "GET", "/api/products/" + product)
                for product in ["0PUK6V6EV0", "1YMWWN1N4O"]
            ]
            tasks.append(pool.submit(order, us))
            values = [f.result() for f in tasks]
            return [
                r
                for value in values
                for r in (value if isinstance(value, list) else [value])
            ]
        selected = [us, foreign] if case == "shipping_slowdown" else [us]
        rows = [r for group in pool.map(order, selected) for r in group]
        if case == "payment_unreachable":
            rows.append(request("GET", "/api/products/0PUK6V6EV0"))
        return rows


async def observations(url, start, end, rows, service):
    client = HttpMcpClient(url)
    traces, logs, errors = [], [], []
    try:
        await client.discover_tools()
        for row in rows:
            offset = 0
            while offset is not None:
                result = await client.call(
                    "get_trace_detail",
                    {
                        "trace_id": row["trace_id"],
                        "limit": 100,
                        "offset": offset,
                    },
                )
                body = result.structuredContent or {}
                if result.isError:
                    errors.append(
                        {
                            "trace_id": row["trace_id"],
                            "error": body.get("error"),
                        }
                    )
                    break
                page = body.get("evidence", [])
                traces.extend(page)
                offset = (
                    page[0]["attributes"]["coverage"]["next_offset"]
                    if page
                    else None
                )
        result = await client.call(
            "search_logs",
            {
                "service": service,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )
        if result.isError:
            errors.append({"service": service, "log_error": True})
        else:
            logs = (result.structuredContent or {}).get("evidence", [])
    finally:
        await client.aclose()
    return {"traces": traces, "logs": logs, "errors": errors}


def reproduced(case, rows, evidence):
    traces, logs = evidence["traces"], evidence["logs"]
    checks = [r for r in rows if r["path"] == "/api/checkout"]
    linked = {e["attributes"]["trace_id"] for e in traces}

    def spans(service, term):
        return [
            e
            for e in traces
            if e.get("service") == service
            and term.lower() in e["attributes"].get("operation", "").lower()
        ]

    def contains(items, text):
        return text.lower() in json.dumps(items).lower()

    conditions = {
        "requests_have_linked_traces": all(
            r["trace_id"] in linked for r in checks
        )
    }
    if case == "payment_failure":
        conditions.update(
            checkout_failed=bool(checks)
            and all(
                r.get("status") == 422
                and "PAYMENT_FAILED" in r.get("error_body", "")
                for r in checks
            ),
            charge_error=any(
                e["attributes"]["is_error"] for e in spans("payment", "charge")
            ),
            application_error=contains(traces + logs, "Invalid token"),
        )
    elif case == "cart_failure":
        groups = {r["user_hash"] for r in rows}
        conditions["same_cart_clear_failure_and_readback"] = all(
            any(
                r["user_hash"] == user
                and r["method"] == "POST"
                and r["path"] == "/api/cart"
                and r.get("status") == 200
                for r in rows
            )
            and any(
                r["user_hash"] == user
                and r["method"] == "DELETE"
                and r.get("status", 0) >= 400
                for r in rows
            )
            and any(
                r["user_hash"] == user and r.get("cart_items", 0) > 0
                for r in rows
            )
            for user in groups
        )
        conditions.update(
            checkout_control=any(r.get("status") == 200 for r in checks),
            emptycart_error=any(
                e["attributes"]["is_error"] for e in spans("cart", "EmptyCart")
            ),
            storage_error=any(
                e["attributes"]["is_error"]
                and (
                    "wasn't able to connect to redis"
                    in e["attributes"]
                    .get("otel_status_description", "")
                    .lower()
                    or "ensureredisconnected"
                    in e["attributes"]
                    .get("otel_status_description", "")
                    .lower()
                )
                for e in spans("cart", "EmptyCart")
            ),
        )
        # Direct DELETE can time out before all spans finish in this build.
        # Successful checkout contains a finished EmptyCart failure; the same
        # hashed cart identity and later GetCart establish residual contents.
        conditions["same_cart_cleanup_and_readback"] = all(
            any(
                e["attributes"]["trace_id"] == row["trace_id"]
                and e["attributes"].get("diagnostic", {}).get("user.id.sha256")
                == user
                and e["attributes"]["is_error"]
                for row in checks
                if row["user_hash"] == user
                for e in spans("cart", "EmptyCart")
            )
            and any(
                e["attributes"]["trace_id"] == row["trace_id"]
                and e["attributes"].get("diagnostic", {}).get("user.id.sha256")
                == user
                and not e["attributes"]["is_error"]
                and e["attributes"]
                .get("diagnostic", {})
                .get("demo.cart.items.count", 0)
                > 0
                for row in rows
                if row["method"] == "GET" and row["user_hash"] == user
                for e in spans("cart", "GetCart")
            )
            for user in groups
        )
    elif case == "shipping_slowdown":
        us = [r for r in checks if r["country"] == "United States"]
        other = [r for r in checks if r["country"] != "United States"]
        conditions.update(
            successful_cohorts=all(r.get("status") == 200 for r in us + other)
            and bool(us and other),
            cohort_delay=bool(us and other)
            and min(r["seconds"] for r in other)
            - max(r["seconds"] for r in us)
            > 3,
            fulfillment_delay=any(
                e["attributes"]["duration_ms"] >= 4500
                for e in spans("shipping", "ship-order")
            ),
            delay_log=contains(logs, "intlShippingSlowdown"),
        )
    elif case == "product_catalog_lock_contention":
        slow = [
            e
            for e in spans("product-catalog", "GetProduct")
            if e["attributes"]["duration_ms"] > 2000
        ]
        slow_ids = {e["attributes"]["trace_id"] for e in slow}
        matching = {
            r["trace_id"]
            for r in rows
            if r["trace_id"] in slow_ids
            and r["seconds"] > 2
            and r.get("status") in {200, 504}
        }
        db_ids = {
            e["attributes"]["trace_id"]
            for e in traces
            if "catalog.products"
            in e["attributes"].get("diagnostic", {}).get("db.query.text", "")
            and e["attributes"].get("diagnostic", {}).get("db.system.name")
            == "postgresql"
        }
        conditions.update(
            table_lock=contains(logs, "ACCESS EXCLUSIVE")
            and contains(logs, "catalog.products"),
            linked_slow_reads=len(matching & db_ids) >= 2,
        )
    elif case == "payment_unreachable":
        conditions.update(
            checkout_failed=bool(checks)
            and all(r.get("status") == 500 for r in checks),
            caller_resolution=contains(
                spans("checkout", "Charge"), "zero addresses"
            ),
            browsing_control=any(
                r["path"].startswith("/api/products/")
                and r.get("status") == 200
                for r in rows
            ),
        )
    return {"reproduced": all(conditions.values()), "conditions": conditions}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=SCENARIO_NAMES, required=True)
    p.add_argument("--demo", type=Path, required=True)
    p.add_argument("--active-flags", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mcp-url", default="http://127.0.0.1:8013/mcp")
    args = p.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt("capture interrupted; restoring flags")

    signal.signal(signal.SIGTERM, interrupted)
    args.output.mkdir(parents=True, exist_ok=False)
    verify_demo_commit(args.demo)
    docker = str(Path.home() / ".docker/bin/docker")
    case = EvaluationCase.load(ROOT / f"evals/cases/{args.case}.json")
    flags = prepare_flag_config(
        args.demo,
        args.output / "fault.flagd.json",
        feature_flag=case.fault.feature_flag,
        variant=case.fault.variant,
    )
    people = json.loads(
        (args.demo / "src/load-generator/people.json").read_text()
    )

    def activate(content):
        args.active_flags.write_bytes(content)
        cmd = compose_command(
            args.demo, ROOT / "evals/compose.payment-failure.yaml", "switch"
        )
        cmd[0] = docker
        subprocess.run(
            cmd,
            cwd=args.demo,
            env=docker_environment(args.active_flags),
            check=True,
            capture_output=True,
            timeout=90,
        )
        subprocess.run(
            [docker, "restart", "--time", "10", "checkout"],
            check=True,
            capture_output=True,
            timeout=90,
        )
        time.sleep(12)

    with Path("/private/tmp/tracemind-demo-evaluation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = args.active_flags.read_bytes()
        (args.output / "original.flagd.json").write_bytes(original)
        try:
            activate(flags.read_bytes())
            start = datetime.now(UTC)
            rows = []
            for batch in range(3):
                rows.extend(traffic(args.case, people))
                print(
                    json.dumps(
                        {
                            "case": args.case,
                            "batch": batch + 1,
                            "requests": len(rows),
                        }
                    ),
                    flush=True,
                )
                time.sleep(15)
            end = datetime.now(UTC)
            time.sleep(15)
            saved = asyncio.run(
                observations(
                    args.mcp_url,
                    start,
                    end,
                    rows,
                    case.ground_truth.root_cause_service,
                )
            )
            save(args.output / "traffic.json", rows)
            save(args.output / "telemetry.json", saved)
            preflight = reproduced(args.case, rows, saved)
            images = subprocess.run(
                [
                    docker,
                    "inspect",
                    "checkout",
                    "payment",
                    "cart",
                    "shipping",
                    "product-catalog",
                    "astronomy-db",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            preflight.update(
                start=start,
                end=end,
                mcp_url=args.mcp_url,
                source_commit="dedc0178918e260823323b8d95005a8cb924b007",
                deployed_images=[
                    {
                        "name": r["Name"],
                        "image_id": r["Image"],
                        "image": r["Config"]["Image"],
                    }
                    for r in json.loads(images.stdout)
                ],
                fault_config_sha256=sha256(flags.read_bytes()).hexdigest(),
            )
            if preflight["reproduced"]:
                save(
                    args.output / "input.json",
                    [
                        {
                            "input": {
                                "incident": case.incident.model_dump(
                                    mode="json"
                                ),
                                "start": start,
                                "end": end,
                            },
                            "expected_output": case.ground_truth.model_dump(
                                mode="json"
                            ),
                        }
                    ],
                )
                preflight["input_sha256"] = sha256(
                    (args.output / "input.json").read_bytes()
                ).hexdigest()
            save(args.output / "preflight.json", preflight)
            print(
                json.dumps({"case": args.case, **preflight}, default=str),
                flush=True,
            )
            if not preflight["reproduced"]:
                raise SystemExit(2)
        finally:
            activate(original)
            save(
                args.output / "restoration.json",
                {
                    "restored": args.active_flags.read_bytes() == original,
                    "original_sha256": sha256(original).hexdigest(),
                    "at": datetime.now(UTC),
                },
            )


if __name__ == "__main__":
    main()
