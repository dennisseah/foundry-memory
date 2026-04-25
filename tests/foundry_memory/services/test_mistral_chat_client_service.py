from unittest.mock import patch

from agent_framework.foundry import FoundryChatClient

from foundry_memory.services.mistral_chat_client_service import (
    MistralChatClientServiceEnv,
    get_client,
)

MODULE = "foundry_memory.services.mistral_chat_client_service"


def _env(model: str = "Mistral-Large-3") -> MistralChatClientServiceEnv:
    return MistralChatClientServiceEnv.model_construct(
        foundry_project_endpoint="https://example.services.ai.azure.com/api/projects/p",
        mistral_deployed_model_name=model,
    )


def test_get_client_passes_expected_params():
    with (
        patch(f"{MODULE}.FoundryChatClient") as mock_cls,
        patch(f"{MODULE}.DefaultAzureCredential") as mock_cred,
    ):
        mock_cred.return_value = "cred-instance"
        sentinel = object()
        mock_cls.return_value = sentinel

        client = get_client(env=_env())

    assert client is sentinel
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["project_endpoint"] == (
        "https://example.services.ai.azure.com/api/projects/p"
    )
    assert kwargs["model"] == "Mistral-Large-3"
    assert kwargs["credential"] == "cred-instance"
    mock_cred.assert_called_once_with()


def test_get_client_uses_custom_model_name():
    with (
        patch(f"{MODULE}.FoundryChatClient") as mock_cls,
        patch(f"{MODULE}.DefaultAzureCredential"),
    ):
        get_client(env=_env(model="Mistral-Small-2"))

    assert mock_cls.call_args.kwargs["model"] == "Mistral-Small-2"


def test_get_client_returns_real_client_type():
    with patch(f"{MODULE}.DefaultAzureCredential"):
        client = get_client(env=_env())
    assert isinstance(client, FoundryChatClient)
