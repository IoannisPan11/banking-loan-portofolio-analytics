"""Public entry point of the analytics assistant.

The tool-call loop is the LangGraph in ``graph.py``:

    scope_guard -> agent <-> tools -> format

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

from .graph import AnalyticsAssistant, AssistantResult, ToolCallingChatModel
from .responses import NO_TOOL_ANSWER
from .tools import ToolRegistry

__all__ = ["NO_TOOL_ANSWER", "AnalyticsAssistant", "AssistantResult", "ToolCallingChatModel", "create_assistant_from_env"]


def create_assistant_from_env(env=None) -> AnalyticsAssistant:
    """Wire the real Databricks repository and Azure OpenAI model from environment variables."""
    from .repository import DatabricksGoldRepository
    from .settings import AzureOpenAISettings, GoldSettings, build_azure_chat_model, load_local_env

    if env is None:
        load_local_env()
    repo = DatabricksGoldRepository(GoldSettings.from_env(env))
    model = build_azure_chat_model(AzureOpenAISettings.from_env(env))
    return AnalyticsAssistant(model, ToolRegistry(repo))
