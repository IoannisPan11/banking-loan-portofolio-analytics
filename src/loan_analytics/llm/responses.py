"""Response assembly shared by the assistant loop and the LangGraph nodes.

Everything here is deterministic: evidence, filters, sources, measure and limitations come from tool results,
and when no tool produced usable evidence the model's text is never used as the answer.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from .models import AnalyticsResponse, ToolResult
from .scope import REFUSAL
from .tools import evidence_for

MAX_QUESTION_CHARS = 1000
NO_TOOL_ANSWER = ("I can only answer from approved BankEasy analytics data, and no approved tool was used for this request, "
                  "so I cannot provide figures for it. I can help with portfolio KPIs, breakdowns by product, branch, segment, "
                  "risk band, channel or month, ageing of open installments, and reference data such as products or risk bands.")


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):   # content blocks
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content).strip()
    return str(content).strip()


def invalid_question_response() -> AnalyticsResponse:
    return AnalyticsResponse(answer="Please ask a question of up to 1000 characters about the loan portfolio.",
                             limitations=["The question was empty or too long; no data was queried."])


def out_of_scope_response(reason: str) -> AnalyticsResponse:
    return AnalyticsResponse(answer=REFUSAL, limitations=[f"Out of scope: {reason}. No data was queried and no action was taken."])


def model_unavailable_response() -> AnalyticsResponse:
    return AnalyticsResponse(answer="The language model is unavailable, so I cannot answer right now.",
                             limitations=["No evidence was retrieved because the language model could not be reached."])


def compose_response(results: list[ToolResult], final_ai: AIMessage | None, exhausted: bool) -> AnalyticsResponse:
    if not results:
        return AnalyticsResponse(answer=NO_TOOL_ANSWER,
                                 limitations=["No approved analytics tool was used, so no BankEasy data is cited."])
    ok = [r for r in results if r.status == "ok"]
    problems = [r for r in results if r.status != "ok"]
    limitations: list[str] = []
    for r in ok:
        limitations += [x for x in r.limitations if x not in limitations]
    for r in problems:
        note = f"{r.tool}: {r.message}" if r.message else f"{r.tool}: no data"
        if note not in limitations:
            limitations.append(note)
    if exhausted:
        limitations.append("The tool-call limit was reached before the model finished; the answer may be incomplete.")

    evidence = [e for r in ok for e in evidence_for(r)]
    sources = list(dict.fromkeys(s for r in ok for s in r.source))
    measures = list(dict.fromkeys(r.measure for r in ok if r.measure))
    if len(ok) == 1:
        filters = dict(ok[0].filters)
    else:
        filters = {f"{r.tool}#{i + 1}": r.filters for i, r in enumerate(ok) if r.filters}

    if not ok:   # nothing usable: never let model text stand in for evidence
        reasons = " ".join(r.message for r in problems if r.message)
        kind = problems[0].error_category if problems else None
        lead = ("No matching Gold data was found" if all(r.status == "empty" for r in problems)
                else "The requested data could not be retrieved from the approved Gold layer")
        answer = f"{lead}. {reasons}".strip()
        if kind == "unsupported":
            answer = f"That measure is not available in the approved semantic layer, so I cannot provide it. {reasons}".strip()
        return AnalyticsResponse(answer=answer, filters=filters, source=sources, measure=None, limitations=limitations)

    answer = text_of(final_ai.content) if final_ai is not None else ""
    if not answer:
        answer = "; ".join(f"{e.label}: {e.value}" for e in evidence[:5]) or "Data retrieved."
    if problems:
        answer += " (Some requested data was unavailable; see limitations.)"
    return AnalyticsResponse(answer=answer, evidence=evidence, filters=filters, source=sources,
                             measure=", ".join(measures) or None, limitations=limitations)
