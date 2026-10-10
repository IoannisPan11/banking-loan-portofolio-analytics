"""Runtime configuration, read from environment variables only (no secrets in code or git).

Gold data access (Databricks SQL warehouse, via the Databricks SDK):
    DATABRICKS_WAREHOUSE_ID   required - no default
    DATABRICKS_CATALOG        default ``data2026acc123456`` (pipelines/README.md)
    DATABRICKS_SCHEMA         default ``team2``             (the team's Gold schema; same default as api/settings.py)
    DATABRICKS_HOST / DATABRICKS_TOKEN (or a profile) are read by the Databricks SDK itself.

Azure OpenAI chat model - all four are required, none has a default:
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION and the deployment name, given as
    AZURE_OPENAI_DEPLOYMENT (preferred) or AZURE_OPENAI_DEPLOYMENT_NAME (accepted alias).

For local development ``load_local_env()`` reads the repository-root ``.env`` (git-ignored). Variables that are
already set in the process environment always win over the file.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

DEFAULT_CATALOG = "data2026acc123456"
DEFAULT_SCHEMA = "team2"


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_local_env(path: Path | None = None) -> bool:
    """Load the repo-root ``.env`` into ``os.environ`` without overriding existing variables.

    Returns True if a file was found. Values are never logged or returned.
    """
    from dotenv import load_dotenv

    env_file = path or REPO_ROOT / ".env"
    return bool(env_file.is_file() and load_dotenv(env_file, override=False))


def _required(env: Mapping[str, str], name: str) -> str:
    value = (env.get(name) or "").strip()
    if not value:
        raise ConfigError(f"Required environment variable {name} is not set.")
    return value


@dataclass(frozen=True)
class GoldSettings:
    warehouse_id: str
    catalog: str = DEFAULT_CATALOG
    schema: str = DEFAULT_SCHEMA
    timeout_seconds: int = 30

    def __post_init__(self):
        for label, value in (("catalog", self.catalog), ("schema", self.schema)):
            if not _IDENTIFIER.match(value):
                raise ConfigError(f"Databricks {label} name is not a valid identifier.")
        if not re.match(r"^[A-Za-z0-9_-]{1,64}$", self.warehouse_id):
            raise ConfigError("DATABRICKS_WAREHOUSE_ID has an unexpected format.")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> GoldSettings:
        env = os.environ if env is None else env
        return cls(
            warehouse_id=_required(env, "DATABRICKS_WAREHOUSE_ID"),
            catalog=(env.get("DATABRICKS_CATALOG") or DEFAULT_CATALOG).strip(),
            schema=(env.get("DATABRICKS_SCHEMA") or DEFAULT_SCHEMA).strip(),
        )


def _deployment(env: Mapping[str, str]) -> str:
    for name in ("AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT_NAME"):
        if (env.get(name) or "").strip():
            return env[name].strip()
    raise ConfigError("Required environment variable AZURE_OPENAI_DEPLOYMENT (or AZURE_OPENAI_DEPLOYMENT_NAME) is not set.")


@dataclass(frozen=True)
class AzureOpenAISettings:
    endpoint: str
    api_key: str
    deployment: str
    api_version: str

    def __repr__(self) -> str:  # never print the key
        return f"AzureOpenAISettings(endpoint={self.endpoint!r}, deployment={self.deployment!r}, api_key=***)"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AzureOpenAISettings:
        env = os.environ if env is None else env
        return cls(
            endpoint=_required(env, "AZURE_OPENAI_ENDPOINT"),
            api_key=_required(env, "AZURE_OPENAI_API_KEY"),
            deployment=_deployment(env),
            api_version=_required(env, "AZURE_OPENAI_API_VERSION"),
        )


def build_azure_chat_model(settings: AzureOpenAISettings):
    """Create the Azure OpenAI chat model (deterministic: temperature 0)."""
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_endpoint=settings.endpoint,
        api_key=settings.api_key,
        azure_deployment=settings.deployment,
        api_version=settings.api_version,
        temperature=0,
    )
