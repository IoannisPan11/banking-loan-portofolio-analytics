import os

from loan_analytics.llm.settings import AzureOpenAISettings, GoldSettings, load_local_env
from loan_analytics.llm.smoke_test import main

BASE = {"AZURE_OPENAI_ENDPOINT": "https://x", "AZURE_OPENAI_API_KEY": "sekret-123", "AZURE_OPENAI_API_VERSION": "v"}


def test_gold_schema_defaults_to_the_teams_schema_and_can_be_overridden():
    assert GoldSettings.from_env({"DATABRICKS_WAREHOUSE_ID": "abc123"}).schema == "team2"
    assert GoldSettings.from_env({"DATABRICKS_WAREHOUSE_ID": "abc123", "DATABRICKS_SCHEMA": "other"}).schema == "other"


def test_deployment_name_alias():
    s = AzureOpenAISettings.from_env({**BASE, "AZURE_OPENAI_DEPLOYMENT_NAME": "dep"})
    assert s.deployment == "dep" and "sekret-123" not in repr(s)


def test_deployment_preferred_over_alias():
    s = AzureOpenAISettings.from_env({**BASE, "AZURE_OPENAI_DEPLOYMENT": "a", "AZURE_OPENAI_DEPLOYMENT_NAME": "b"})
    assert s.deployment == "a"


def test_load_local_env_does_not_override(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("LLM_TEST_A=file\nLLM_TEST_B=file\n")
    monkeypatch.setenv("LLM_TEST_A", "process")
    monkeypatch.delenv("LLM_TEST_B", raising=False)
    assert load_local_env(f)
    assert os.environ["LLM_TEST_A"] == "process" and os.environ["LLM_TEST_B"] == "file"
    monkeypatch.delenv("LLM_TEST_B")


def test_smoke_test_reports_config_error(monkeypatch, capsys):
    for k in ("DATABRICKS_WAREHOUSE_ID", "AZURE_OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("loan_analytics.llm.smoke_test.load_local_env", lambda: False)
    monkeypatch.setattr("loan_analytics.llm.settings.load_local_env", lambda: False)   # create_assistant_from_env loads .env too
    assert main(["hi"]) == 2
    assert "CONFIG ERROR" in capsys.readouterr().err
