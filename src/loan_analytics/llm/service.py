"""The tool-call loop of the analytics assistant.

    SystemMessage + HumanMessage
      -> chat_model.bind_tools([...]).invoke(messages) -> AIMessage
           no tool call  -> safe response (nothing is answered from model memory)
           tool call(s)  -> validate typed args -> execute the approved function -> ToolMessage(tool_call_id=...)
      -> invoke again with the full history ... -> final AIMessage
      -> AnalyticsResponse: evidence / filters / source / measure / limitations come from the tool results

Safety properties enforced by code (not left to the prompt):
* out-of-scope action or SQL requests are refused before the model is called;
* unknown tools, malformed arguments and unsupported measures are never executed;
* if no tool produced evidence, the answer is a deterministic "unavailable" message - model text is not used;
* every step is recorded in a deterministic trace (no secrets, no payloads).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from .models import AnalyticsResponse, ToolResult, TraceEntry
from .prompts import build_messages
from .scope import REFUSAL, detect_out_of_scope
from .tools import ToolRegistry, evidence_for

log = logging.getLogger("loan_analytics.llm")

MAX_QUESTION_CHARS = 1000
NO_TOOL_ANSWER = ("I can only answer from approved BankEasy analytics data, and no approved tool was used for this request, "
                  "so I cannot provide figures for it. I can help with portfolio KPIs, breakdowns by product, branch, segment, "
                  "risk band, channel or month, ageing of open installments, and reference data such as products or risk bands.")


class ToolCallingChatModel(Protocol):
    def bind_tools(self, tools: list, **kwargs: Any): ...


@dataclass
class AssistantResult:
    """Structured response plus the full audit trail."""

    response: AnalyticsResponse
    trace: list[TraceEntry]
    trace_id: str
    question: str
    messages: list[BaseMessage] = field(default_factory=list)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):   # content blocks
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content).strip()
    return str(content).strip()


class AnalyticsAssistant:
    def __init__(self, chat_model: ToolCallingChatModel, registry: ToolRegistry, max_iterations: int = 4):
        self._model = chat_model
        self._registry = registry
        self._max_iterations = max_iterations

    # ---- public ------------------------------------------------------------------------------------------------
    def ask(self, question: str) -> AssistantResult:
        trace_id = uuid.uuid4().hex[:12]
        trace: list[TraceEntry] = []
        question = (question or "").strip()

        def finish(response: AnalyticsResponse, messages: list[BaseMessage]) -> AssistantResult:
            log.info("assistant_done trace_id=%s steps=%d measure=%s sources=%s", trace_id, len(trace), response.measure,
                     ",".join(response.source))
            return AssistantResult(response=response, trace=trace, trace_id=trace_id, question=question, messages=messages)

        if not question or len(question) > MAX_QUESTION_CHARS:
            trace.append(TraceEntry(step="input_validation", status="refused", error_category="invalid_question"))
            return finish(AnalyticsResponse(answer="Please ask a question of up to 1000 characters about the loan portfolio.",
                                            limitations=["The question was empty or too long; no data was queried."]), [])

        reason = detect_out_of_scope(question)
        if reason:
            trace.append(TraceEntry(step="scope_guard", status="refused", error_category="out_of_scope"))
            log.info("assistant_refused trace_id=%s reason=%s", trace_id, reason)
            return finish(AnalyticsResponse(answer=REFUSAL, limitations=[f"Out of scope: {reason}. No data was queried and no action was taken."]), [])

        messages = build_messages(question, self._registry.summary())
        bound = self._model.bind_tools(self._registry.langchain_tools())
        results: list[ToolResult] = []
        final_ai: AIMessage | None = None
        exhausted = False

        for iteration in range(1, self._max_iterations + 1):
            try:
                ai = bound.invoke(messages)
            except Exception as exc:  # noqa: BLE001 - model/service outage must not produce guesses
                trace.append(TraceEntry(step="model_call", status="error", error_category=type(exc).__name__))
                log.warning("assistant_model_error trace_id=%s error=%s", trace_id, type(exc).__name__)
                return finish(AnalyticsResponse(answer="The language model is unavailable, so I cannot answer right now.",
                                                limitations=["No evidence was retrieved because the language model could not be reached."]), messages)
            messages.append(ai)
            trace.append(TraceEntry(step="model_call", status="ok", tool=",".join(c["name"] for c in ai.tool_calls) or None))
            if not ai.tool_calls:
                final_ai = ai
                break
            for call in ai.tool_calls:
                name, args, call_id = call["name"], call.get("args"), call.get("id") or uuid.uuid4().hex
                ex = self._registry.execute(name, args)
                r = ex.result
                results.append(r)
                trace.append(TraceEntry(step="tool_call", tool=name, tool_call_id=call_id,
                                        raw_args=args if isinstance(args, dict) else None, validated_args=ex.validated_args,
                                        sources=r.source, status=r.status, error_category=r.error_category,
                                        row_count=r.row_count, duration_ms=round(ex.duration_ms, 2)))
                log.info("assistant_tool trace_id=%s tool=%s status=%s category=%s rows=%s ms=%.1f", trace_id, name, r.status,
                         r.error_category, r.row_count, ex.duration_ms)
                messages.append(ToolMessage(content=r.for_model(), tool_call_id=call_id, name=name,
                                            status="error" if r.status == "error" else "success"))
        else:
            exhausted = True

        return finish(self._compose(results, final_ai, exhausted), messages)

    # ---- response assembly ---------------------------------------------------------------------------------------
    @staticmethod
    def _compose(results: list[ToolResult], final_ai: AIMessage | None, exhausted: bool) -> AnalyticsResponse:
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

        answer = _text(final_ai.content) if final_ai is not None else ""
        if not answer:
            answer = "; ".join(f"{e.label}: {e.value}" for e in evidence[:5]) or "Data retrieved."
        if problems:
            answer += " (Some requested data was unavailable; see limitations.)"
        return AnalyticsResponse(answer=answer, evidence=evidence, filters=filters, source=sources,
                                 measure=", ".join(measures) or None, limitations=limitations)


def create_assistant_from_env(env=None) -> AnalyticsAssistant:
    """Wire the real Databricks repository and Azure OpenAI model from environment variables."""
    from .repository import DatabricksGoldRepository
    from .settings import AzureOpenAISettings, GoldSettings, build_azure_chat_model, load_local_env

    if env is None:
        load_local_env()
    repo = DatabricksGoldRepository(GoldSettings.from_env(env))
    model = build_azure_chat_model(AzureOpenAISettings.from_env(env))
    return AnalyticsAssistant(model, ToolRegistry(repo))
