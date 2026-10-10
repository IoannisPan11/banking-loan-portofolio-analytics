"""Manual end-to-end check of the real assistant (Azure OpenAI -> tool -> Databricks Gold -> final answer).

    uv run python -m loan_analytics.llm.smoke_test "What is the current approval rate?"
    uv run python -m loan_analytics.llm.smoke_test --schema team2 "Show the approval rate by product"

Configuration comes from the environment / repo-root ``.env`` (see ``settings.py``). Output is a safe trace
(tool, validated arguments, sources, status) plus the structured AnalyticsResponse - never credentials, SQL
or raw payloads.
"""
from __future__ import annotations

import argparse
import os
import sys

from .errors import ConfigError
from .service import create_assistant_from_env
from .settings import load_local_env


def format_trace(result) -> str:
    lines = [f"trace_id: {result.trace_id}"]
    for t in result.trace:
        parts = [f"step={t.step}", f"status={t.status}"]
        if t.tool:
            parts.append(f"tool={t.tool}")
        if t.validated_args is not None:
            parts.append(f"validated_args={t.validated_args}")
        if t.sources:
            parts.append(f"sources={t.sources}")
        if t.row_count is not None:
            parts.append(f"rows={t.row_count}")
        if t.error_category:
            parts.append(f"error={t.error_category}")
        lines.append("  " + " ".join(parts))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", help="natural-language question for the analytics assistant")
    parser.add_argument("--schema", help="override DATABRICKS_SCHEMA (e.g. team2)")
    args = parser.parse_args(argv)

    load_local_env()
    if args.schema:
        os.environ["DATABRICKS_SCHEMA"] = args.schema
    try:
        assistant = create_assistant_from_env()
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - never echo messages that may carry secrets
        print(f"SETUP ERROR: {type(exc).__name__}", file=sys.stderr)
        return 2

    result = assistant.ask(args.question)
    print(f"Question: {result.question}\n")
    print(format_trace(result))
    print("\nAnalyticsResponse:")
    print(result.response.model_dump_json(indent=2))
    model_failed = any(t.step == "model_call" and t.status == "error" for t in result.trace)
    return 1 if model_failed else 0


if __name__ == "__main__":
    sys.exit(main())
