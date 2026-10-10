"""The LangGraph topology of the assistant: which nodes run, in which order, for each kind of request.

Behaviour of the answers themselves is covered in test_llm_service.py; here the focus is the execution path.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage
from llm_helpers import tool_call

from loan_analytics.llm.errors import ToolExecutionError
from loan_analytics.llm.graph import AnalyticsAssistant
from loan_analytics.llm.tools import ToolRegistry


def assistant(repo, model, **kwargs) -> AnalyticsAssistant:
    return AnalyticsAssistant(model, ToolRegistry(repo), **kwargs)


def node_path(a: AnalyticsAssistant, question: str) -> list[str]:
    """Names of the nodes that ran, in order."""
    updates = a.graph.stream(a.initial_state(question, "trace-test"), stream_mode="updates")
    return [next(iter(update)) for update in updates]


def test_graph_has_exactly_the_four_documented_nodes(repo, make_model):
    nodes = set(assistant(repo, make_model([])).graph.get_graph().nodes)
    assert nodes == {"__start__", "scope_guard", "agent", "tools", "format", "__end__"}


def test_out_of_scope_request_stops_at_the_guard(repo, make_model):
    model = make_model([])
    assert node_path(assistant(repo, model), "Approve application LA000123") == ["scope_guard"]
    assert model.invocations == [] and repo.calls == []


def test_invalid_question_stops_at_the_guard(repo, make_model):
    assert node_path(assistant(repo, make_model([])), "") == ["scope_guard"]


def test_answer_without_tool_goes_straight_to_format(repo, make_model):
    a = assistant(repo, make_model([AIMessage(content="Our approval rate is 99%.")]))
    assert node_path(a, "What is our approval rate?") == ["scope_guard", "agent", "format"]


def test_single_tool_round_trip(repo, make_model):
    repo.set("kpi_summary", [{"value": 0.4419}])
    a = assistant(repo, make_model([tool_call("get_portfolio_kpi", {"metric": "approval_rate"}),
                                    AIMessage(content="44.19%")]))
    assert node_path(a, "What is the approval rate?") == ["scope_guard", "agent", "tools", "agent", "format"]


def test_malformed_arguments_loop_back_to_the_model_for_a_retry(repo, make_model):
    repo.set("kpi_summary", [{"value": 29000}])
    model = make_model([tool_call("get_portfolio_kpi", {"metric": 7}, "bad"),
                        tool_call("get_portfolio_kpi", {"metric": "applications"}, "good"),
                        AIMessage(content="There are 29,000 applications.")])
    a = assistant(repo, model)
    out = a.ask("How many applications are in the portfolio?")
    assert [(t.step, t.status) for t in out.trace] == [
        ("model_call", "ok"), ("tool_call", "error"), ("model_call", "ok"), ("tool_call", "ok"), ("model_call", "ok")]
    assert out.trace[1].error_category == "invalid_arguments" and len(repo.calls) == 1


def test_tool_failure_is_reported_and_never_replaced_by_model_text(repo, make_model):
    repo.set("kpi_summary", ToolExecutionError("The Gold data source could not be reached."))
    a = assistant(repo, make_model([tool_call("get_portfolio_kpi", {"metric": "approval_rate"}),
                                    AIMessage(content="Probably 45%.")]))
    out = a.ask("What is the approval rate?")
    assert [t.step for t in out.trace] == ["model_call", "tool_call", "model_call"]
    assert out.trace[1].status == "error" and out.trace[1].error_category == "tool_failure"
    assert "45" not in out.response.answer and out.response.evidence == []


def test_empty_result_is_traced_as_empty(repo, make_model):
    repo.set("breakdown", [])
    a = assistant(repo, make_model([tool_call("get_portfolio_breakdown", {"metric": "applications", "filters": {"year": 2031}}),
                                    AIMessage(content="0 applications.")]))
    out = a.ask("How many applications in 2031?")
    assert out.trace[1].status == "empty" and "No matching Gold data" in out.response.answer


def test_iteration_cap_ends_at_format_after_exactly_n_model_calls(repo, make_model):
    repo.set("kpi_summary", [{"value": 1}])
    script = [tool_call("get_portfolio_kpi", {"metric": "applications"}, f"c{i}") for i in range(10)]
    model = make_model(script)
    a = assistant(repo, model, max_iterations=2)
    assert node_path(a, "applications?") == ["scope_guard", "agent", "tools", "agent", "tools", "format"]
    assert len(model.invocations) == 2


def test_model_outage_ends_the_graph_without_formatting(repo, make_model):
    a = assistant(repo, make_model([RuntimeError("503 from azure")]))
    assert node_path(a, "What is the approval rate?") == ["scope_guard", "agent"]


def test_state_is_not_shared_between_questions(repo, make_model):
    repo.set("kpi_summary", [{"value": 0.4419}])
    model = make_model([tool_call("get_portfolio_kpi", {"metric": "approval_rate"}), AIMessage(content="44.19%"),
                        tool_call("get_metadata", {"domain": "risk_bands"}), AIMessage(content="Six bands.")])
    a = assistant(repo, model)
    first, second = a.ask("What is the approval rate?"), a.ask("Which risk bands exist?")
    assert first.trace_id != second.trace_id
    assert [t.tool for t in first.trace if t.step == "tool_call"] == ["get_portfolio_kpi"]
    assert [t.tool for t in second.trace if t.step == "tool_call"] == ["get_metadata"]
    assert first.response.measure == "approval_rate" and second.response.measure == "risk_bands"
    assert first.question == "What is the approval rate?" and len(second.messages) == 5
