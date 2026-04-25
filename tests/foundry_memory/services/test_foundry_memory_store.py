from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import HttpResponseError

from foundry_memory.services.foundry_memory_store import (
    FoundryMemoryStore,
    FoundryMemoryStoreEnv,
)

MODULE = "foundry_memory.services.foundry_memory_store"


def _env() -> FoundryMemoryStoreEnv:
    return FoundryMemoryStoreEnv.model_construct(
        foundry_project_endpoint="https://example.services.ai.azure.com/api/projects/p",
        memory_store_chat_model_deployment_name="gpt-test",
        memory_store_embedding_model_deployment_name="emb-test",
    )


@pytest.fixture
def store():
    with (
        patch(f"{MODULE}.AIProjectClient") as mock_pc_cls,
        patch(f"{MODULE}.DefaultAzureCredential"),
    ):
        mock_pc = MagicMock()
        mock_pc_cls.return_value = mock_pc
        s = FoundryMemoryStore(env=_env())
    return s, mock_pc


def test_init_builds_definition_and_project_client():
    with (
        patch(f"{MODULE}.AIProjectClient") as mock_pc_cls,
        patch(f"{MODULE}.DefaultAzureCredential") as mock_cred,
        patch(f"{MODULE}.MemoryStoreDefaultDefinition") as mock_def,
        patch(f"{MODULE}.MemoryStoreDefaultOptions") as mock_opts,
    ):
        mock_cred.return_value = "cred"
        mock_opts.return_value = "options"
        mock_def.return_value = "definition"

        s = FoundryMemoryStore(env=_env())

        mock_opts.assert_called_once_with(
            chat_summary_enabled=True, user_profile_enabled=True
        )
        mock_def.assert_called_once_with(
            chat_model="gpt-test",
            embedding_model="emb-test",
            options="options",
        )
        mock_pc_cls.assert_called_once_with(
            endpoint="https://example.services.ai.azure.com/api/projects/p",
            credential="cred",
        )
        assert s.definition == "definition"
        assert s.project_client is mock_pc_cls.return_value


def test_create_memory_store_success(store):
    s, mock_pc = store
    mock_pc.beta.memory_stores.create.return_value = "ms-instance"

    s.create_memory_store("name1", "desc1")

    mock_pc.beta.memory_stores.create.assert_called_once_with(
        name="name1", definition=s.definition, description="desc1"
    )
    assert s.memory_store == "ms-instance"


def test_create_memory_store_already_exists_ignored(store):
    s, mock_pc = store
    mock_pc.beta.memory_stores.create.side_effect = HttpResponseError(
        message="memory store already exists"
    )

    s.create_memory_store("name1", "desc1", ignore_if_exists=True)


def test_create_memory_store_already_exists_raises_when_not_ignored(store):
    s, mock_pc = store
    mock_pc.beta.memory_stores.create.side_effect = HttpResponseError(
        message="memory store already exists"
    )

    with pytest.raises(HttpResponseError):
        s.create_memory_store("name1", "desc1", ignore_if_exists=False)


def test_create_memory_store_other_error_always_raises(store):
    s, mock_pc = store
    mock_pc.beta.memory_stores.create.side_effect = HttpResponseError(message="boom")

    with pytest.raises(HttpResponseError):
        s.create_memory_store("name1", "desc1", ignore_if_exists=True)


def test_update_memory_store_calls_poller(store):
    s, mock_pc = store
    poller = MagicMock()
    mock_pc.beta.memory_stores.begin_update_memories.return_value = poller

    items = [{"role": "user", "content": "hi"}]
    s.update_memory_store("name1", "user1", items)  # type: ignore[arg-type]

    mock_pc.beta.memory_stores.begin_update_memories.assert_called_once_with(
        name="name1", scope="user1", items=items, update_delay=0
    )
    poller.result.assert_called_once_with()


def test_search_memories_returns_contents(store):
    s, mock_pc = store
    mem1 = MagicMock()
    mem1.memory_item.content = "content-1"
    mem2 = MagicMock()
    mem2.memory_item.content = "content-2"
    results = MagicMock()
    results.memories = [mem1, mem2]
    mock_pc.beta.memory_stores.search_memories.return_value = results

    out = s.search_memories("name1", "user1", "query text")

    assert out == ["content-1", "content-2"]
    mock_pc.beta.memory_stores.search_memories.assert_called_once_with(
        name="name1", scope="user1", items="query text"
    )


def test_search_memories_returns_none_on_error(store, capsys):
    s, mock_pc = store
    mock_pc.beta.memory_stores.search_memories.side_effect = RuntimeError("nope")

    out = s.search_memories("name1", "user1", "query")

    assert out is None
    assert "Error searching memories" in capsys.readouterr().out


def test_delete_memory_store_success(store):
    s, mock_pc = store
    s.delete_memory_store("name1")
    mock_pc.beta.memory_stores.delete.assert_called_once_with(name="name1")


def test_delete_memory_store_swallows_http_error(store, capsys):
    s, mock_pc = store
    mock_pc.beta.memory_stores.delete.side_effect = HttpResponseError(message="boom")

    s.delete_memory_store("name1")

    assert "Error deleting memory store" in capsys.readouterr().out
