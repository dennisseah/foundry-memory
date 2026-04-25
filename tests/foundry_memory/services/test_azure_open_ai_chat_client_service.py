from unittest.mock import patch

from agent_framework.openai import OpenAIChatCompletionClient

from foundry_memory.services.azure_open_ai_chat_client_service import (
    AzureOpenAIChatClientServiceEnv,
    get_client,
)

MODULE = "foundry_memory.services.azure_open_ai_chat_client_service"


def _env(api_key: str | None = None) -> AzureOpenAIChatClientServiceEnv:
    return AzureOpenAIChatClientServiceEnv.model_construct(
        azure_openai_endpoint="https://example.openai.azure.com",
        azure_openai_deployed_model_name="gpt-test",
        azure_openai_api_version="2024-10-21",
        azure_openai_api_key=api_key,
    )


def test_get_client_with_api_key():
    with patch(f"{MODULE}.OpenAIChatCompletionClient") as mock_cls:
        sentinel = object()
        mock_cls.return_value = sentinel

        client = get_client(env=_env(api_key="secret"))

    assert client is sentinel
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["azure_endpoint"] == "https://example.openai.azure.com"
    assert kwargs["model"] == "gpt-test"
    assert kwargs["api_version"] == "2024-10-21"
    assert kwargs["api_key"] == "secret"
    assert "credential" not in kwargs


def test_get_client_uses_default_credential_when_no_api_key():
    with (
        patch(f"{MODULE}.OpenAIChatCompletionClient") as mock_cls,
        patch(f"{MODULE}.DefaultAzureCredential") as mock_cred,
    ):
        mock_cred.return_value = "cred-instance"
        sentinel = object()
        mock_cls.return_value = sentinel

        client = get_client(env=_env(api_key=None))

    assert client is sentinel
    kwargs = mock_cls.call_args.kwargs
    assert "api_key" not in kwargs
    assert kwargs["credential"] == "cred-instance"
    mock_cred.assert_called_once_with()


def test_get_client_returns_real_client_type():
    """Smoke test wiring through to the real OpenAIChatCompletionClient."""
    with patch(f"{MODULE}.DefaultAzureCredential"):
        client = get_client(env=_env(api_key=None))
    assert isinstance(client, OpenAIChatCompletionClient)
