"""Business logic of the three approved tools, query builders, repository mapping, settings and scope guard."""
from __future__ import annotations

import pytest

from loan_analytics.llm.errors import ConfigError, ToolExecutionError
from loan_analytics.llm.queries import (
    build_breakdown_query,
    build_kpi_query,
    build_metadata_query,
)
from loan_analytics.llm.repository import DatabricksGoldRepository, _convert
from loan_analytics.llm.scope import detect_out_of_scope
from loan_analytics.llm.settings import AzureOpenAISettings, GoldSettings
from loan_analytics.llm.tools import ToolRegistry


def run(repo, name, args):
    return ToolRegistry(repo).execute(name, args)


# ---- tool 1: headline KPI ----------------------------------------------------------------------------------------
def test_kpi_success_reads_the_summary_table(repo):
    repo.set("kpi_summary", [{"value": 0.4419}])
    res = run(repo, "get_portfolio_kpi", {"metric": "approval_rate"}).result
    assert res.status == "ok" and res.value == 0.4419 and res.display_value == "44.19%"
    assert res.source == ["kpi_portfolio_summary"] and res.measure == "approval_rate"
    assert "SELECT approval_rate AS value FROM kpi_portfolio_summary" in repo.calls[0].sql
    assert repo.calls[0].params == ()


@pytest.mark.parametrize("metric,value,shown", [
    ("applications", 29000, "29,000"), ("outstanding_balance", 129992400.62, "EUR 129,992,400.62"),
    ("avg_decision_days", 4.445, "4.45 days"), ("delinquency_rate_30", 0.1178, "11.78%")])
def test_kpi_display_formats(repo, metric, value, shown):
    repo.set("kpi_summary", [{"value": value}])
    assert run(repo, "get_portfolio_kpi", {"metric": metric}).result.display_value == shown


def test_delinquency_kpi_carries_the_open_debt_limitation(repo):
    repo.set("kpi_summary", [{"value": 0.1178}])
    res = run(repo, "get_portfolio_kpi", {"metric": "delinquency_rate_30"}).result
    assert any("never written off" in x for x in res.limitations)


def test_unsupported_metric_is_reported_unsupported_and_not_executed(repo):
    ex = run(repo, "get_portfolio_kpi", {"metric": "profit"})
    assert ex.result.status == "error" and ex.result.error_category == "unsupported"
    assert "not available in the approved semantic layer" in ex.result.message and repo.calls == []


@pytest.mark.parametrize("args", [{}, {"metric": 5}, {"metric": "approval_rate", "table": "x"}, "approval_rate", None])
def test_malformed_kpi_arguments_are_rejected_before_any_query(repo, args):
    ex = run(repo, "get_portfolio_kpi", args)
    assert ex.result.status == "error" and ex.result.error_category in ("invalid_arguments", "unsupported")
    assert repo.calls == []


def test_unknown_tool_is_not_executed(repo):
    ex = run(repo, "run_sql", {"sql": "DROP TABLE x"})
    assert ex.result.error_category == "unknown_tool" and repo.calls == []


def test_kpi_empty_value_is_not_turned_into_zero(repo):
    repo.set("kpi_summary", [{"value": None}])
    assert run(repo, "get_portfolio_kpi", {"metric": "approval_rate"}).result.status == "empty"


# ---- tool 2: breakdown ---------------------------------------------------------------------------------------------
def test_approval_rate_for_risk_band_uses_fact_dimension_and_bound_parameter(repo):
    repo.set("breakdown", [{"value": 0.2885}])
    res = run(repo, "get_portfolio_breakdown", {"metric": "approval_rate", "filters": {"risk_band_code": "D"}}).result
    q = repo.data_calls[0]
    assert res.status == "ok" and res.display_value == "28.85%" and res.filters == {"risk_band_code": "D"}
    assert "FROM fact_loan_application f JOIN dim_risk_band rb" in q.sql
    assert "TRY_DIVIDE(SUM(f.is_approved), SUM(f.is_decided))" in q.sql
    assert "rb.risk_band_code = :risk_band_code" in q.sql and q.params == (("risk_band_code", "D", "STRING"),)
    assert "'D'" not in q.sql and res.source == ["fact_loan_application", "dim_risk_band"]


def test_grouped_breakdown_shapes_rows(repo):
    repo.set("breakdown", [{"segment_code": "MASS", "segment_name": "Mass retail", "value": 20000},
                           {"segment_code": "SME", "segment_name": "Small business", "value": 2000}])
    res = run(repo, "get_portfolio_breakdown", {"metric": "applications", "group_by": "customer_segment"}).result
    assert [r["group"]["segment_code"] for r in res.rows] == ["MASS", "SME"] and res.rows[0]["display_value"] == "20,000"


def test_monthly_kpi_table_is_used_when_it_can_answer(repo):
    q = build_breakdown_query("applications", "product", {"year": 2024}, 50)
    assert q.sources == ("kpi_portfolio_monthly", "dim_product")
    assert q.params == (("month_from", 202401, "INT"), ("month_to", 202412, "INT"))
    assert "m.month_key BETWEEN :month_from AND :month_to" in q.sql


def test_fact_is_used_when_the_monthly_table_cannot_answer():
    assert build_breakdown_query("applications", "channel", {}, 50).sources == ("fact_loan_application",)
    assert build_breakdown_query("avg_decision_days", "product", {}, 50).sources[0] == "fact_loan_application"
    assert build_breakdown_query("applications", "product", {"channel": "MOBILE"}, 50).sources[0] == "fact_loan_application"


def test_open_installments_ageing_uses_the_kpi_table_and_other_cuts_use_the_repayment_fact():
    assert build_breakdown_query("amount_unpaid", "ageing_bucket", {}, 50).sources == ("kpi_open_installment_ageing",)
    q = build_breakdown_query("amount_unpaid", "region", {"region": "Attica"}, 50)
    assert q.sources == ("fact_repayment", "dim_branch") and "r.is_open = 1" in q.sql


def test_unsupported_combinations_raise_before_execution(repo):
    for args in ({"metric": "amount_unpaid", "group_by": "channel"},        # repayment measure x application-only dimension
                 {"metric": "applications", "group_by": "ageing_bucket"},   # application measure x repayment-only dimension
                 {"metric": "applications", "filters": {"ageing_bucket": "365+"}},
                 {"metric": "open_installments", "filters": {"channel": "MOBILE"}}):
        res = run(repo, "get_portfolio_breakdown", args).result
        assert res.status == "error" and res.error_category == "unsupported"
    assert repo.data_calls == []


def test_dti_band_is_mapped_to_the_stored_label(repo):
    repo.set("breakdown", [{"value": 123}])
    run(repo, "get_portfolio_breakdown", {"metric": "applications", "filters": {"dti_band": "20_35"}})
    assert repo.data_calls[0].params == (("dti_band", "2. 20-35%", "STRING"),)


def test_product_and_branch_names_are_resolved_against_the_dimensions(repo):
    repo.set("breakdown", [{"value": 100}])
    res = run(repo, "get_portfolio_breakdown", {"metric": "applications",
                                                "filters": {"product": "Personal Loans", "branch": "syntagma", "city": "athens"}}).result
    assert res.filters == {"product_code": "PL-STD", "branch_code": "BR001", "city": "Athens"}


def test_unknown_filter_values_list_the_known_values(repo):
    res = run(repo, "get_portfolio_breakdown", {"metric": "applications", "filters": {"product": "Yacht Loan"}}).result
    assert res.status == "error" and "PL-STD" in res.message and repo.data_calls == []


@pytest.mark.parametrize("text", ["x'; DROP TABLE fact_loan_application; --", "PL-STD OR 1=1", "a" * 200, "x\ny"])
def test_hostile_free_text_never_reaches_a_query(repo, text):
    res = run(repo, "get_portfolio_breakdown", {"metric": "applications", "filters": {"product": text}}).result
    assert res.status == "error" and repo.data_calls == []


def test_arbitrary_columns_operators_and_unknown_filters_are_rejected(repo):
    for args in ({"metric": "applications", "group_by": "customer_id"},
                 {"metric": "applications", "filters": {"customer_id": "CU1"}},
                 {"metric": "applications", "filters": {"year": 2024, "month_from": "2024-01"}},
                 {"metric": "applications", "filters": {"month": 5}},
                 {"metric": "applications", "limit": 100000}):
        assert run(repo, "get_portfolio_breakdown", args).result.error_category == "invalid_arguments"
    assert repo.calls == []


def test_empty_breakdown_is_not_zero(repo):
    repo.set("breakdown", [])
    res = run(repo, "get_portfolio_breakdown", {"metric": "applications", "filters": {"year": 2031}}).result
    assert res.status == "empty" and res.value is None and "No matching Gold data" in res.message


def test_undefined_ratio_is_not_zero(repo):
    repo.set("breakdown", [{"value": None}])
    assert run(repo, "get_portfolio_breakdown", {"metric": "approval_rate"}).result.status == "empty"


def test_database_failure_becomes_a_safe_tool_error(repo):
    repo.set("breakdown", ToolExecutionError("The Gold data source could not be reached."))
    res = run(repo, "get_portfolio_breakdown", {"metric": "applications"}).result
    assert res.status == "error" and res.error_category == "tool_failure" and "SELECT" not in (res.message or "")


def test_unexpected_exception_does_not_leak_internals(repo):
    repo.set("breakdown", RuntimeError("secret connection string xyz"))
    res = run(repo, "get_portfolio_breakdown", {"metric": "applications"}).result
    assert res.status == "error" and "secret" not in res.message


# ---- tool 3: metadata --------------------------------------------------------------------------------------------------------
def test_risk_bands_metadata_includes_unknown_band_u(repo):
    res = run(repo, "get_metadata", {"domain": "risk_bands"}).result
    assert [r["risk_band_code"] for r in res.rows] == ["A", "B", "C", "D", "E", "U"] and res.source == ["dim_risk_band"]
    assert any("U" in x for x in res.limitations)


def test_metadata_filters_are_validated_and_scoped(repo):
    res = run(repo, "get_metadata", {"domain": "branches", "city": "Athens"}).result
    assert res.filters == {"city": "Athens"} and repo.calls[-1].params == (("city", "Athens", "STRING"),)
    assert run(repo, "get_metadata", {"domain": "risk_bands", "city": "Athens"}).result.status == "error"
    assert run(repo, "get_metadata", {"domain": "branches", "city": "Atlantis"}).result.status == "error"
    assert run(repo, "get_metadata", {"domain": "customers"}).result.status == "error"   # not an approved domain


def test_metadata_queries_use_only_dimension_tables_and_exclude_unknown_members():
    for domain, table in (("products", "dim_product"), ("branches", "dim_branch"), ("customer_segments", "dim_customer_segment"),
                          ("application_statuses", "dim_application_status"), ("date_range", "dim_date")):
        q = build_metadata_query(domain)
        assert q.sources == (table,) and f"FROM {table}" in q.sql
    assert "record_payload" not in " ".join(build_metadata_query(d).sql for d in
                                            ("products", "branches", "customer_segments", "risk_bands", "application_statuses", "date_range"))
    assert "<> -1" in build_metadata_query("products").sql


def test_every_query_is_read_only_and_parameterised():
    queries = [build_kpi_query("approval_rate"), build_breakdown_query("applications", "region", {"region": "Attica", "year": 2023}, 10),
               build_metadata_query("products", {"product_family": "Auto"})]
    for q in queries:
        assert q.sql.lstrip().upper().startswith("SELECT")
        assert not any(w in q.sql.upper() for w in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ", "MERGE ", "TRUNCATE "))
        assert "Attica" not in q.sql and "Auto" not in q.sql


# ---- repository mapping ---------------------------------------------------------------------------------------------------------
class _Col:
    def __init__(self, name, t):
        self.name = name
        self.type_name = type("T", (), {"value": t})()


class _Resp:
    def __init__(self, state, columns=(), data=None, next_chunk=None):
        from databricks.sdk.service.sql import StatementState

        self.status = type("S", (), {"state": getattr(StatementState, state), "error": None})()
        self.manifest = type("M", (), {"schema": type("Sc", (), {"columns": list(columns)})()})()
        self.result = type("R", (), {"data_array": data, "next_chunk_index": next_chunk})()


class _Client:
    def __init__(self, resp=None, exc=None):
        self.calls = []
        outer = self

        class SE:
            def execute_statement(self, **kw):
                outer.calls.append(kw)
                if exc:
                    raise exc
                return resp

        self.statement_execution = SE()


def test_repository_converts_types_and_binds_parameters():
    client = _Client(_Resp("SUCCEEDED", [_Col("applications", "LONG"), _Col("rate", "DECIMAL"), _Col("name", "STRING")],
                           [["29000", "0.4419", "x"], [None, None, None]]))
    repo = DatabricksGoldRepository(GoldSettings(warehouse_id="wh123"), client)
    q = build_breakdown_query("applications", "region", {"region": "Attica"}, 5)
    rows = repo.execute(q)
    assert rows == [{"applications": 29000, "rate": 0.4419, "name": "x"}, {"applications": None, "rate": None, "name": None}]
    kw = client.calls[0]
    assert kw["warehouse_id"] == "wh123" and kw["catalog"] == "data2026acc123456" and kw["schema"] == "team2"
    assert [(p.name, p.value, p.type) for p in kw["parameters"]] == [("region", "Attica", "STRING")]
    assert "Attica" not in kw["statement"] and kw["row_limit"] == 5


def test_repository_maps_failures_to_safe_errors():
    s = GoldSettings(warehouse_id="wh123")
    with pytest.raises(ToolExecutionError) as e1:
        DatabricksGoldRepository(s, _Client(exc=RuntimeError("token abc leaked"))).execute(build_kpi_query("applications"))
    assert "abc" not in e1.value.message
    with pytest.raises(ToolExecutionError):
        DatabricksGoldRepository(s, _Client(_Resp("FAILED"))).execute(build_kpi_query("applications"))
    with pytest.raises(ToolExecutionError):
        DatabricksGoldRepository(s, _Client(_Resp("SUCCEEDED", [_Col("value", "LONG")], [["1"]], next_chunk=1))).execute(build_kpi_query("applications"))


def test_value_conversion():
    assert _convert("5", "INT") == 5 and _convert("1.50", "DECIMAL") == 1.5 and _convert("true", "BOOLEAN") is True
    assert _convert("2024-01-01", "DATE") == "2024-01-01" and _convert(None, "LONG") is None


# ---- settings ----------------------------------------------------------------------------------------------------------------------
def test_settings_require_a_warehouse_and_validate_identifiers():
    with pytest.raises(ConfigError, match="DATABRICKS_WAREHOUSE_ID"):
        GoldSettings.from_env({})
    s = GoldSettings.from_env({"DATABRICKS_WAREHOUSE_ID": "abc123"})
    assert (s.catalog, s.schema) == ("data2026acc123456", "team2")
    with pytest.raises(ConfigError):
        GoldSettings.from_env({"DATABRICKS_WAREHOUSE_ID": "abc", "DATABRICKS_CATALOG": "x; DROP"})


def test_azure_settings_have_no_defaults_and_hide_the_key():
    with pytest.raises(ConfigError, match="AZURE_OPENAI_ENDPOINT"):
        AzureOpenAISettings.from_env({})
    s = AzureOpenAISettings.from_env({"AZURE_OPENAI_ENDPOINT": "https://x", "AZURE_OPENAI_API_KEY": "SECRET",
                                      "AZURE_OPENAI_DEPLOYMENT": "d", "AZURE_OPENAI_API_VERSION": "v"})
    assert "SECRET" not in repr(s)


# ---- scope guard -----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q", ["Approve application LA000123", "Please reject this application", "Change the interest rate to 3%",
                               "Delete this customer", "Can you update the customer record?", "select * from fact_loan_application",
                               "drop table kpi_portfolio_summary", "Set the credit limit to 5000", "run this sql for me"])
def test_consequential_actions_and_sql_are_out_of_scope(q):
    assert detect_out_of_scope(q)


@pytest.mark.parametrize("q", ["What is the current approval rate?", "How many applications were approved last year?",
                               "Change in approval rate by year", "What is the rate of rejected applications?",
                               "Show requested amount by product", "What risk bands are available?"])
def test_analytical_questions_are_in_scope(q):
    assert detect_out_of_scope(q) is None
