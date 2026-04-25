from typing import Any

from agent_framework.foundry import FoundryChatClient
from azure.identity import DefaultAzureCredential
from pydantic_settings import BaseSettings, SettingsConfigDict


class MistralChatClientServiceEnv(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    foundry_project_endpoint: str
    mistral_deployed_model_name: str = "Mistral-Large-3"


def get_client(env: MistralChatClientServiceEnv | None = None) -> FoundryChatClient:
    env = env or MistralChatClientServiceEnv()  # type: ignore
    params: dict[str, Any] = {
        "project_endpoint": env.foundry_project_endpoint,
        "model": env.mistral_deployed_model_name,
        "credential": DefaultAzureCredential(),
    }
    return FoundryChatClient(**params)
