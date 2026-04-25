from typing import Any

from agent_framework.openai import OpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureOpenAIChatClientServiceEnv(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    azure_openai_endpoint: str
    azure_openai_deployed_model_name: str
    azure_openai_api_version: str
    azure_openai_api_key: str | None = None


def get_client(
    env: AzureOpenAIChatClientServiceEnv | None = None,
) -> OpenAIChatCompletionClient:
    env = env or AzureOpenAIChatClientServiceEnv()  # type: ignore
    params: dict[str, Any] = {
        "azure_endpoint": env.azure_openai_endpoint,
        "model": env.azure_openai_deployed_model_name,
        "api_version": env.azure_openai_api_version,
    }

    if env.azure_openai_api_key:
        params["api_key"] = env.azure_openai_api_key
    else:
        params["credential"] = DefaultAzureCredential()

    return OpenAIChatCompletionClient(**params)
