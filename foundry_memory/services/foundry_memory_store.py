from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    MemoryStoreDefaultDefinition,
    MemoryStoreDefaultOptions,
)
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential
from openai.types.responses import ResponseInputParam
from pydantic_settings import BaseSettings, SettingsConfigDict


class FoundryMemoryStoreEnv(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    foundry_project_endpoint: str
    memory_store_chat_model_deployment_name: str
    memory_store_embedding_model_deployment_name: str


class FoundryMemoryStore:
    def __init__(self, env: FoundryMemoryStoreEnv | None = None):
        self.env = env or FoundryMemoryStoreEnv()  # type: ignore

        self.definition = MemoryStoreDefaultDefinition(
            chat_model=self.env.memory_store_chat_model_deployment_name,
            embedding_model=self.env.memory_store_embedding_model_deployment_name,
            options=MemoryStoreDefaultOptions(
                chat_summary_enabled=True, user_profile_enabled=True
            ),
        )
        self.project_client = AIProjectClient(
            endpoint=self.env.foundry_project_endpoint,
            credential=DefaultAzureCredential(),
        )

    def create_memory_store(
        self, name: str, description: str, ignore_if_exists: bool = False
    ):
        """Create memory store

        Args:
            name (str): name of the store
            description (str): description of the store
            ignore_if_exists (bool, optional): ignore if the memory store already
            exists. Defaults to False.

        Raises:
            e (HttpResponseError): when the memory store creation fails due to an HTTP
            error
        """
        try:
            self.memory_store = self.project_client.beta.memory_stores.create(
                name=name, definition=self.definition, description=description
            )
        except HttpResponseError as e:
            if not ignore_if_exists or "already exists" not in str(e):
                raise e

    def update_memory_store(self, name: str, scope: str, items: ResponseInputParam):
        """Update memory store

        Args:
            name (str): name of the memory store to update
            scope (str): scope of the memory store to update
            items (ResponseInputParam): items to update in the memory store
        """
        update_poller = self.project_client.beta.memory_stores.begin_update_memories(
            name=name, scope=scope, items=items, update_delay=0
        )
        update_poller.result()

    def search_memories(self, name: str, scope: str, query: str) -> list[str] | None:
        """Search memories in a memory store

        Args:
            name (str): name of the memory store to search
            scope (str): scope of the memory store to search
            query (str): query string to search for in the memory store

        Returns:
            list[str] | None: result of the search, a list of memory contents matching
            the query,
        """
        try:
            results = self.project_client.beta.memory_stores.search_memories(
                name=name,
                scope=scope,
                items=query,
            )
            return [mem.memory_item.content for mem in results.memories]
        except Exception as e:
            print(f"Error searching memories in store {name}: {e}")
            return None

    def delete_memory_store(self, name: str):
        """Delete memory store.

        Args:
            name (str): name of the memory store to delete
        """
        try:
            self.project_client.beta.memory_stores.delete(name=name)
        except HttpResponseError as e:
            print(f"Error deleting memory store {name}: {e}")
