#!/usr/bin/env python3
"""Run a fixed Qwen4-Flash-Next coherence corpus over OpenAI-compatible APIs.

The runner is deliberately endpoint-only: it does not load or unload models.
That keeps Windows LM Studio lifecycle control separate from the Mac's mainline
OMLX installation and makes Q4, Q8, OpenRouter, and experimental MLX results
directly comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_CORPUS = Path(__file__).with_name("qwen4_flash_next_eval_corpus.jsonl")
RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class EndpointHTTPError(RuntimeError):
    def __init__(self, status: int, url: str, detail: str, retry_after: float | None):
        super().__init__(f"HTTP {status} from {url}: {detail}")
        self.status = status
        self.retry_after = retry_after


def load_dotenv(path: Path) -> None:
    """Load a small .env without printing or overwriting existing secrets."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def api_url(base_url: str, route: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/" + route.lstrip("/")


def request_json(
    method: str,
    url: str,
    *,
    api_key: str | None = None,
    body: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": "next48-eval/1"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        retry_after = None
        try:
            retry_after = float(exc.headers.get("Retry-After", ""))
        except (TypeError, ValueError):
            pass
        raise EndpointHTTPError(exc.code, url, detail, retry_after) from exc


def request_json_with_retries(
    method: str,
    url: str,
    *,
    retries: int,
    retry_backoff: float,
    **kwargs: Any,
) -> dict[str, Any]:
    """Retry transient endpoint failures without hiding permanent errors."""
    for attempt in range(retries + 1):
        try:
            return request_json(method, url, **kwargs)
        except EndpointHTTPError as exc:
            if exc.status not in RETRYABLE_HTTP_STATUS or attempt == retries:
                raise
            delay = exc.retry_after
            if delay is None:
                delay = min(30.0, retry_backoff * (2**attempt))
            print(
                f"  transient HTTP {exc.status}; retry "
                f"{attempt + 1}/{retries} in {delay:g}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def endpoint_settings(args) -> tuple[str | None, str]:
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set (checked --env-file too)")
    model = args.model or (os.environ.get(args.model_env) if args.model_env else None)
    if not model:
        raise RuntimeError("model is required via --model or --model-env")
    return api_key, model


def load_corpus(path: Path) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        item = json.loads(raw)
        item_id = item.get("id")
        if not item_id or item_id in seen:
            raise ValueError(f"{path}:{line_number}: missing or duplicate id {item_id!r}")
        if not isinstance(item.get("messages"), list):
            raise ValueError(f"{path}:{line_number}: messages must be a list")
        seen.add(item_id)
        records.append(item)
    return records


def response_text(payload: dict[str, Any]) -> tuple[str, str | None]:
    message = payload["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    return str(content), str(reasoning) if reasoning is not None else None


def evaluate_checks(text: str, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # OpenAI-compatible reasoning parsers commonly leave the two separator
    # newlines at the start of final content.  They are transport/template
    # framing, not a failure to obey an "answer only" instruction.
    text = text.strip()
    outcomes = []
    for check in checks:
        kind = check["type"]
        passed = False
        detail = None
        if kind == "regex":
            passed = re.search(check["pattern"], text, re.IGNORECASE | re.DOTALL) is not None
        elif kind == "not_regex":
            passed = re.search(check["pattern"], text, re.IGNORECASE | re.DOTALL) is None
        elif kind == "contains_all":
            folded = text.casefold()
            missing = [value for value in check["values"] if value.casefold() not in folded]
            passed, detail = not missing, {"missing": missing}
        elif kind == "json_equals":
            try:
                parsed = json.loads(text.removeprefix("```json").removesuffix("```").strip())
                passed = parsed == check["value"]
                detail = {"parsed": parsed}
            except (ValueError, TypeError) as exc:
                detail = {"error": str(exc)}
        else:
            raise ValueError(f"unknown check type {kind!r}")
        outcomes.append({"type": kind, "passed": passed, "detail": detail})
    return outcomes


def run(args) -> None:
    load_dotenv(args.env_file)
    api_key, model = endpoint_settings(args)
    corpus = load_corpus(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")

    run_id = f"{args.endpoint}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    passed_items = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, item in enumerate(corpus, 1):
            body = {
                "model": model,
                "messages": item["messages"],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": item.get("max_tokens", args.max_tokens),
                "stream": False,
            }
            if args.enable_thinking is not None:
                body["chat_template_kwargs"] = {
                    "enable_thinking": args.enable_thinking
                }
            if args.seed is not None:
                body["seed"] = args.seed
            started = time.perf_counter()
            record = {
                "schema_version": 1,
                "run_id": run_id,
                "endpoint": args.endpoint,
                "base_url": args.base_url,
                "model": model,
                "item_id": item["id"],
                "category": item.get("category"),
                "request": body,
                "created_utc": datetime.now(UTC).isoformat(),
            }
            try:
                payload = request_json_with_retries(
                    "POST",
                    api_url(args.base_url, "chat/completions"),
                    retries=args.retries,
                    retry_backoff=args.retry_backoff,
                    api_key=api_key,
                    body=body,
                    timeout=args.timeout,
                )
                text, reasoning = response_text(payload)
                outcomes = evaluate_checks(text, item.get("checks", []))
                item_passed = bool(outcomes) and all(check["passed"] for check in outcomes)
                passed_items += int(item_passed)
                record.update(
                    {
                        "latency_seconds": round(time.perf_counter() - started, 6),
                        "response": text,
                        "reasoning": reasoning,
                        "finish_reason": payload["choices"][0].get("finish_reason"),
                        "usage": payload.get("usage"),
                        "checks": outcomes,
                        "passed": item_passed,
                        "error": None,
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "latency_seconds": round(time.perf_counter() - started, 6),
                        "response": None,
                        "reasoning": None,
                        "checks": [],
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index:02d}/{len(corpus):02d}] {item['id']}: "
                f"{'PASS' if record['passed'] else 'FAIL'} "
                f"({record['latency_seconds']:.2f}s)",
                flush=True,
            )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "output": str(args.output.resolve()),
                "passed": passed_items,
                "total": len(corpus),
            },
            indent=2,
        )
    )


def probe(args) -> None:
    load_dotenv(args.env_file)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    payload = request_json(
        "GET", api_url(args.base_url, "models"), api_key=api_key, timeout=args.timeout
    )
    models = [item.get("id") for item in payload.get("data", [])]
    print(json.dumps({"endpoint": args.endpoint, "models": models}, indent=2))


def summarize(args) -> None:
    corpus_by_id = None
    if args.recheck:
        corpus_by_id = {item["id"]: item for item in load_corpus(args.corpus)}
    rows = []
    for path in args.inputs:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if corpus_by_id is not None:
            for record in records:
                item = corpus_by_id[record["item_id"]]
                checks = evaluate_checks(record.get("response") or "", item.get("checks", []))
                record["passed"] = bool(checks) and all(check["passed"] for check in checks)
        rows.append(
            {
                "file": str(path),
                "endpoint": records[0].get("endpoint") if records else None,
                "model": records[0].get("model") if records else None,
                "passed": sum(bool(record.get("passed")) for record in records),
                "total": len(records),
                "errors": sum(record.get("error") is not None for record in records),
                "mean_latency_seconds": (
                    sum(float(record["latency_seconds"]) for record in records) / len(records)
                    if records else None
                ),
                "failed_items": [record["item_id"] for record in records if not record.get("passed")],
            }
        )
    print(json.dumps(rows, indent=2))


def add_endpoint_args(parser) -> None:
    parser.add_argument("--endpoint", required=True, help="Stable label stored in results")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key-env", help="Environment variable containing the API key")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--timeout", type=float, default=600.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe_parser = subparsers.add_parser("probe")
    add_endpoint_args(probe_parser)
    probe_parser.set_defaults(func=probe)

    run_parser = subparsers.add_parser("run")
    add_endpoint_args(run_parser)
    run_parser.add_argument("--model")
    run_parser.add_argument("--model-env")
    run_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--max-tokens", type=int, default=512)
    run_parser.add_argument("--seed", type=int, default=7)
    run_parser.add_argument("--retries", type=int, default=4)
    run_parser.add_argument("--retry-backoff", type=float, default=2.0)
    thinking_group = run_parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="Explicitly enable reasoning in chat-template kwargs",
    )
    thinking_group.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Explicitly disable reasoning in chat-template kwargs",
    )
    run_parser.set_defaults(enable_thinking=None)
    run_parser.add_argument("--overwrite", action="store_true")
    run_parser.set_defaults(func=run)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("inputs", type=Path, nargs="+")
    summary_parser.add_argument(
        "--recheck",
        action="store_true",
        help="Re-evaluate raw responses with the current corpus checks",
    )
    summary_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    summary_parser.set_defaults(func=summarize)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
