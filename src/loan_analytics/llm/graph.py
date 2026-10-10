"""LangGraph orchestration of the analytics assistant.

    START -> scope_guard -> agent <-> tools -> format -> END

    scope_guard : empty/oversized or out-of-scope requests are refused before the model is called
    agent       : the chat model (tools bound) decides: answer, or request approved tool call(s)
    tools       : validate the typed arguments, execute the approved function, append a ToolMessage per call
    format      : build the structured response from the tool results (never from model text alone)

The graph only orchestrates; messages, tools, prompts and validation stay in langchain-core and the
modules next to this one. ``AnalyticsAssistant.ask`` returns an ``AssistantResult``: the structured response plus the full audit trail.
"""
from __future__ import annotations

import logging
import operator
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .models import AnalyticsResponse, ToolResult, TraceEntry
from .prompts import build_messages
from .responses import (
    MAX_QUESTION_CHARS,
    compose_response,
    invalid_question_response,
    model_unavailable_response,
    out_of_scope_response,
)
from .scope import detect_out_of_scope
from .tools import ToolRegistry

log = logging.getLogger("loan_analytics.llm")


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


class AssistantState(TypedDict, total=False):
    """State carried between nodes. Lists with a reducer are appended to; all other keys are overwritten."""

    # input
    question: str
    trace_id: str

    # conversation (SystemMessage, HumanMessage, AIMessage, ToolMessage) - appended by the add_messages reducer
    messages: Annotated[list[AnyMessage], add_messages]

    # audit: every executed tool result and every step, in order - appended by each node
    results: Annotated[list[ToolResult], operator.add]
    trace: Annotated[list[TraceEntry], operator.add]

    # loop control
    iteration: int                  # number of model calls so far; the loop stops at the configured maximum
    final_ai: AIMessage | None      # the model's last message when it answered without a tool call
    exhausted: bool                 # the iteration cap was hit before the model finished

    # output, set by a terminal node (scope_guard refusal, model failure or format)
    response: AnalyticsResponse


class AnalyticsAssistant:
    """The tool-calling assistant, orchestrated as a LangGraph."""

    def __init__(self, chat_model: ToolCallingChatModel, registry: ToolRegistry, max_iterations: int = 4):
        self._model = chat_model
        self._registry = registry
        self._max_iterations = max_iterations
        self._bound = None   # chat model with the approved tools bound, created on first use
        self.graph = self._build()

    # ---- nodes --------------------------------------------------------------------------------------------------
    def _scope_guard(self, state: AssistantState) -> dict[str, Any]:
        question = state["question"]
        if not question or len(question) > MAX_QUESTION_CHARS:
            return {"response": invalid_question_response(),
                    "trace": [TraceEntry(step="input_validation", status="refused", error_category="invalid_question")]}
        reason = detect_out_of_scope(question)
        if reason:
            log.info("assistant_refused trace_id=%s reason=%s", state["trace_id"], reason)
            return {"response": out_of_scope_response(reason),
                    "trace": [TraceEntry(step="scope_guard", status="refused", error_category="out_of_scope")]}
        return {"messages": build_messages(question, self._registry.summary())}

    def _agent(self, state: AssistantState) -> dict[str, Any]:
        if self._bound is None:
            self._bound = self._model.bind_tools(self._registry.langchain_tools())
        iteration = state.get("iteration", 0) + 1
        try:
            ai = self._bound.invoke(state["messages"])
        except Exception as exc:  # noqa: BLE001 - model/service outage must not produce guesses
            log.warning("assistant_model_error trace_id=%s error=%s", state["trace_id"], type(exc).__name__)
            return {"iteration": iteration, "response": model_unavailable_response(),
                    "trace": [TraceEntry(step="model_call", status="error", error_category=type(exc).__name__)]}
        entry = TraceEntry(step="model_call", status="ok", tool=",".join(c["name"] for c in ai.tool_calls) or None)
        return {"messages": [ai], "iteration": iteration, "trace": [entry], "final_ai": None if ai.tool_calls else ai}

    def _tools(self, state: AssistantState) -> dict[str, Any]:
        trace_id = state["trace_id"]
        messages: list[ToolMessage] = []
        results: list[ToolResult] = []
        trace: list[TraceEntry] = []
        for call in state["messages"][-1].tool_calls:
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
        return {"messages": messages, "results": results, "trace": trace,
                "exhausted": state.get("iteration", 0) >= self._max_iterations}

    def _format(self, state: AssistantState) -> dict[str, Any]:
        return {"response": compose_response(state.get("results", []), state.get("final_ai"), state.get("exhausted", False))}

    # ---- edges --------------------------------------------------------------------------------------------------
    @staticmethod
    def _after_guard(state: AssistantState) -> str:
        return END if "response" in state else "agent"

    @staticmethod
    def _after_agent(state: AssistantState) -> str:
        if "response" in state:          # model unavailable
            return END
        return "tools" if state["messages"][-1].tool_calls else "format"

    @staticmethod
    def _after_tools(state: AssistantState) -> str:
        return "format" if state.get("exhausted") else "agent"

    def _build(self):
        g = StateGraph(AssistantState)
        g.add_node("scope_guard", self._scope_guard)
        g.add_node("agent", self._agent)
        g.add_node("tools", self._tools)
        g.add_node("format", self._format)
        g.add_edge(START, "scope_guard")
        g.add_conditional_edges("scope_guard", self._after_guard, ["agent", END])
        g.add_conditional_edges("agent", self._after_agent, ["tools", "format", END])
        g.add_conditional_edges("tools", self._after_tools, ["agent", "format"])
        g.add_edge("format", END)
        return g.compile()

    # ---- public -------------------------------------------------------------------------------------------------
    @staticmethod
    def initial_state(question: str, trace_id: str) -> AssistantState:
        return {"question": question, "trace_id": trace_id, "iteration": 0, "exhausted": False, "results": [], "trace": []}

    def ask(self, question: str) -> AssistantResult:
        trace_id = uuid.uuid4().hex[:12]
        question = (question or "").strip()
        state = self.graph.invoke(self.initial_state(question, trace_id))
        response, trace = state["response"], state["trace"]
        log.info("assistant_done trace_id=%s steps=%d measure=%s sources=%s", trace_id, len(trace), response.measure,
                 ",".join(response.source))
        return AssistantResult(response=response, trace=trace, trace_id=trace_id, question=question,
                               messages=state.get("messages", []))
