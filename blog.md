---
title: Building Memory-Aware Agents with Azure AI Foundry Memory Store
description:
  A hands-on walkthrough of using the Azure AI Foundry Memory Store to give chat
  agents persistent, scoped, and searchable memory.
author: Dennis Seah
ms.date: 2026-04-25
ms.topic: how-to
keywords:
  - azure ai foundry
  - memory store
  - agent framework
  - rag
  - chat memory
estimated_reading_time: 8
---

## Building Memory-Aware Agents with Azure AI Foundry Memory Store

Most chat agents are amnesiacs. Each session starts from a blank slate, and any
context the user shared yesterday (preferences, decisions, half-finished plans)
has to be re-supplied or stuffed back into the prompt. The new Azure AI Foundry
Memory Store offers a cleaner answer: a managed, scoped, searchable memory that
your agent can read from and write to as a first-class capability.

This post walks through what the Memory Store gives you, how it behaves in
practice, and the patterns that worked well while wiring it into a real agent
built on the Microsoft Agent Framework.

[!NOTE] The Memory Store is exposed via `AIProjectClient.beta.memory_stores`.
The "beta" namespace is a reminder that the surface area can still evolve, but
the core operations (create, update, search, delete) are stable enough to build
on.

### What is the Foundry Memory Store?

The Foundry Memory Store is a managed component inside an Azure AI Foundry
project that stores conversational memory for your agents. You hand it message
turns; it handles summarization, embedding, and retrieval on your behalf, backed
by a chat model and an embedding model that you nominate when you create the
store. You can store additional information such as user profile data,
preferences, or any structured metadata that you want your agent to recall later
alongside the raw conversation turns.

In code, defining a store looks like this:

```python
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    MemoryStoreDefaultDefinition,
    MemoryStoreDefaultOptions,
)
from azure.identity import DefaultAzureCredential

definition = MemoryStoreDefaultDefinition(
    chat_model="gpt-4o-mini",                  # used for summarization
    embedding_model="text-embedding-3-large",  # used for semantic search
    options=MemoryStoreDefaultOptions(
        chat_summary_enabled=True,
        user_profile_enabled=True,
    ),
)

project_client = AIProjectClient(
    endpoint="<your-foundry-project-endpoint>",
    credential=DefaultAzureCredential(),
)

project_client.beta.memory_stores.create(
    name="memory-prod",
    definition=definition,
    description="Production chat memory",
)
```

Two things are worth highlighting here:

- `chat_summary_enabled` lets the store compress long conversations into salient
  summaries instead of replaying every turn back to your agent.
- `user_profile_enabled` lets the store accumulate longer-lived facts about a
  user across sessions (preferences, recurring topics, working context)
  separately from raw turn-by-turn dialogue.

### Memory is scoped, and that scope is where the power is

Every read and write into the store carries a `scope`. The scope is just a
string, but it is the lever that lets a single memory store serve very different
access patterns:

- `scope="user-123"` for per-user memory in a chat assistant.
- `scope="tenant-acme"` for per-tenant memory in a multi-tenant SaaS.
- `scope="project-orion"` for per-project memory in a planning agent.
- `scope="user-123/project-orion"` for composite scopes that combine the two.

Because the scope is supplied at call time rather than baked into the store's
definition, you do not need a separate store per user or tenant. One store, many
scopes, and isolation is enforced on the read/write path:

```python
project_client.beta.memory_stores.begin_update_memories(
    name="memory-prod",
    scope=f"user-{user_id}",
    items=turns,        # list of message params
    update_delay=0,
).result()
```

This is the single most useful property of the Memory Store in practice. It maps
cleanly onto multi-user assistants, agent workflows with per-project context,
and even hierarchical setups where a user inherits broader tenant-level facts.

### Memory is searchable, and search is fast

When your agent needs context, you do not pull the entire memory blob. You ask a
question:

```python
results = project_client.beta.memory_stores.search_memories(
    name="memory-prod",
    scope=f"user-{user_id}",
    items="What has the user said about their deployment region?",
)
recalled = [m.memory_item.content for m in results.memories]
```

The store returns the top relevant memory items for that scope. In practice,
this call is consistently snappy, well within the latency budget of a normal
chat turn. That means you can comfortably run a search **on every user message**
as a lightweight RAG step, then inject the recalled items into your system
prompt:

```python
instructions = SYSTEM_PROMPT
if recalled:
    instructions += (
        "\n\nThe following is recalled context about the user from prior "
        "conversations. Treat these as established facts and use them to "
        "interpret short or ambiguous follow-up questions in continuity "
        "with what was previously discussed:\n"
        + "\n".join(f"- {m}" for m in recalled)
    )
agent = Agent(chat_client, instructions=instructions)
```

This pattern keeps your prompt small (only what is relevant to _this_ turn lands
in context) while still giving the agent a long-term memory it can lean on.

### Updates are slower, and that is fine

The asymmetry between reads and writes is the one behavior worth designing
around. `search_memories` is fast; `begin_update_memories` is, as the name
implies, a long-running operation. The store is doing real work on your behalf,
including summarizing turns, extracting profile facts, and embedding content,
and that takes longer than a synchronous in-memory append would.

Two practical implications:

1. **Do not block the user on the write.** After the agent has produced its
   response and you have shown it to the user, hand the new turns to the store
   and let the update happen in the background. The next turn can still read the
   latest snapshot the store has finalized; you do not need this update to land
   before responding again.
2. **Batch where it makes sense.** You can hand the store both the user message
   and the assistant response in a single update call, which is both cheaper and
   more semantically coherent than two separate writes:

   ```python
   memory_store.update_memory_store(
       name=MEM_STORE_NAME,
       scope=user_id,
       items=[
           EasyInputMessageParam(content=user_msg,      role="user",      type="message"),
           EasyInputMessageParam(content=assistant_msg, role="assistant", type="message"),
       ],
   )
   ```

> [!TIP] Treat `update_memory_store` like a write to a search index, not like an
> in-memory list append. Fire-and-forget after each turn, and rely on
> `search_memories` to surface what the model needs.

### Putting it together: a thin wrapper

A small wrapper around the four operations (create, update, search, delete)
keeps the rest of the codebase clean and gives you a single place to handle the
quirks: idempotent create, swallowing benign delete errors, and returning `None`
on search failures so the agent degrades gracefully:

```python
class FoundryMemoryStore:
    def create_memory_store(self, name, description, ignore_if_exists=False): ...
    def update_memory_store(self, name, scope, items): ...
    def search_memories(self, name, scope, query) -> list[str] | None: ...
    def delete_memory_store(self, name): ...
```

The agent loop then becomes refreshingly boring:

1. On user input, call `search_memories(scope=user_id, query=user_msg)`.
2. Build the agent with recalled memory injected into its instructions.
3. Stream the response back to the user.
4. Call `update_memory_store(scope=user_id, items=[user_msg, assistant_msg])`
   and move on.

### Why this matters

The Foundry Memory Store closes a gap that previously required a fair amount of
glue: a vector database for semantic recall, a separate summarizer for long
histories, a scheme for per-user isolation, and orchestration code to keep them
all in sync. With the Memory Store you get all of that as a managed primitive,
with scope as a clean isolation boundary and a search path fast enough to use on
every turn.

The takeaways from building on it:

- Decide on the natural isolation boundary first (user, tenant, or project) and
  bake that into every read and write from day one.
- Read on every turn and write asynchronously. Search is cheap, updates are not.
  Design the latency budget around that.
- Let the store do the summarization. Enable `chat_summary_enabled` and
  `user_profile_enabled` and stop hand-rolling that logic in your prompt.

For agents that need to feel like they actually know the user across sessions,
days, and topics, the Foundry Memory Store is a meaningful step up from "stuff
everything into the system prompt and hope for the best."

## Demo

In order to demonstrate how the Foundry Memory Store works in practice, we'll
walk through a simple example where an agent interacts with a user, recalls
relevant context from previous turns, and updates the memory store.

We have these setup in the demo environment:

- Two users, `jondoe` and `maryann`. Our intention is to show the memory store
  maintaining separate scopes for each user so that the agent can recall context
  relevant to the current user without leaking information between them.
- Two large language models, `gpt-5-mini` and `Mistral-Large-3` so that we can
  show how the agent can be swapped between different LLMs while still
  leveraging the same memory store for context recall.

Here is what we going to do in the demo:

1. We start with user `jondoe` and the agent is `gpt-5-mini`. Since this is the
   beginning of the conversation, the memory store for `jondoe` is empty, so the
   agent will have no prior context to recall.
2. `jondoe` sends a message to the agent, "What is the pricing model for Azure
   Blob Storage?". The agent receives this message and responds accordingly. The
   user and assistant messages from this turn are then written to the memory
   store under the scope for `jondoe`, so that future turns can recall this
   context when relevant.
3. We switch to user `maryann` and the agent remains `gpt-5-mini`. Since this is
   the first turn for `maryann`, the memory store for her scope is empty as
   well, so the agent will again have no prior context to recall for this user.
   This demonstrates that the memory store correctly isolates context by user
   scope.
4. `maryann` sends a message to the agent, "Can I have Python 3.13 runtime for
   my Azure Functions app?". The agent receives this message and responds
   accordingly. The user and assistant messages from this turn are then written
   to the memory store under the scope for `maryann`, so that future turns can
   recall this context when relevant.
5. We switch the user back to `jondoe`. This time, we can see that the memory
   store for `jondoe` contains the previous turn where he asked about the
   pricing model for Azure Blob Storage. We switch the agent to
   `Mistral-Large-3` and send a new message from `jondoe`, "What is the pricing
   model for Azure Cosmos DB?". The agent receives this message and responds
   accordingly. The user and assistant messages from this turn are then written
   to the memory store under the scope for `jondoe`, so that future turns can
   recall this context when relevant.
6. We switch back to `maryann`. At this point, the memory store for `maryann`
   contains the previous turn where she asked about the Python runtime for Azure
   Functions. This illustrates that the memory store has correctly maintained
   separate scopes for each user: the agent can now respond to `maryann` with
   context from her previous turn without ever seeing the context from
   `jondoe`'s conversation.
7. We switch back to `jondoe`. At this point, the memory store for `jondoe`
   contains both of his previous turns: the question about the pricing models
   for Azure Blob Storage and Cosmos DB. This demonstrates that the memory store
   correctly accumulates context for each user over multiple turns, allowing the
   agent to recall relevant information from earlier in the conversation when
   responding to subsequent messages from the same user.
8. Lastly, `jondoe` asks `What are the resources that I enquired previously?`.
   The agent can now look into the memory store under the scope for `jondoe` and
   retrieve the previous turns. The agent responds with previously asked
   questions and responses.

This demonstration shows how the memory store can maintain separate scopes for
different users while allowing the agent to accumulate context over multiple
turns for each user independently. It also illustrates how the agent can be
swapped between different LLMs while still leveraging the same memory store for
context recall, ensuring that the conversation history is preserved and relevant
context is available to the agent regardless of which model is currently
handling the conversation.

Microsoft Agent Framework is used to build this demo, providing the convenience
of managing multiple LLMs, handling user sessions, and integrating with the
memory store so that context can be maintained and recalled seamlessly across
different users and turns in the conversation.
