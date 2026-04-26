"""Streamlit chat UI backed by the Foundry memory store."""

import asyncio
import os
import signal

import streamlit as st
from agent_framework import Agent
from openai.types.responses import EasyInputMessageParam

from foundry_memory.services.azure_open_ai_chat_client_service import (
    get_client as get_azure_openai_client,
)
from foundry_memory.services.foundry_memory_store import FoundryMemoryStore
from foundry_memory.services.mistral_chat_client_service import (
    get_client as get_mistral_client,
)

PROVIDER_OPTIONS = ["OpenAI (gpt-5-mini)", "Mistral (Mistral-Large-3)"]
DEFAULT_PROVIDER = "OpenAI (gpt-5-mini)"
DEFAULT_USER_ID = "jondoe"
user_id = DEFAULT_USER_ID
USER_OPTIONS = ["jondoe", "maryann"]

SYSTEM_PROMPT = (
    "You are a helpful assistant that helps to answer questions around Microsoft Azure."
    "Any services mentioned should be treated as part of the Azure ecosystem and "
    "should be interpreted in the context of Azure services and offerings."
)
MEM_STORE_NAME = "memory8"


st.set_page_config(page_title="Foundry Memory Chat", page_icon="💬")

st.markdown(
    """
    <style>
    html, body, [class*="st-"], .stMarkdown, .stChatMessage,
    .stTextInput, .stSelectbox, .stButton, .stCaption {
        font-size: 0.85rem !important;
    }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.05rem !important; }
    .stChatMessage p { font-size: 0.85rem !important; }
    code, pre { font-size: 0.78rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_memory_store() -> FoundryMemoryStore:
    store = FoundryMemoryStore()
    store.create_memory_store(
        MEM_STORE_NAME, f"{MEM_STORE_NAME} description", ignore_if_exists=True
    )
    return store


def build_agent(provider: str) -> Agent:
    with st.spinner("Recalling memory..."):
        recalled = memory_store.search_memories(
            name=MEM_STORE_NAME,
            scope=user_id,
            query="user's recent topics, preferences, and ongoing questions",
        )
    st.session_state["recalled"] = recalled or []
    previous_memory = st.session_state["recalled"]

    fn_client = (
        get_azure_openai_client if provider.startswith("OpenAI") else get_mistral_client
    )
    chat_client = fn_client()
    instructions = SYSTEM_PROMPT
    if previous_memory:
        instructions += (
            "\n\nThe following is recalled context about the user from prior "
            "conversations. Treat these as established facts and use them to "
            "interpret short or ambiguous follow-up questions in continuity "
            "with what was previously discussed:\n"
            + "\n".join(f"- {m}" for m in previous_memory)
        )
    return Agent(chat_client, instructions=instructions)  # type: ignore


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return a persistent event loop for this Streamlit session."""
    loop = st.session_state.get("_loop")
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        st.session_state["_loop"] = loop
    asyncio.set_event_loop(loop)
    return loop


def run_async[T](coro) -> T:  # type: ignore[type-arg]
    """Run an async coroutine from Streamlit's sync context."""
    return _get_loop().run_until_complete(coro)


def stream_agent_text(agent: Agent, prompt: str, session) -> "tuple[object, list[str]]":
    """Return a sync generator of text deltas plus a list that captures the full reply.

    Yields strings as they arrive from the agent's streaming response.
    The captured reply is appended to ``collected`` so the caller can read the
    final text after ``st.write_stream`` consumes the generator.
    """
    loop = _get_loop()
    stream = agent.run(prompt, session=session, stream=True)
    aiter_ = stream.__aiter__()
    collected: list[str] = []

    def gen():
        try:
            while True:
                try:
                    update = loop.run_until_complete(aiter_.__anext__())
                except StopAsyncIteration:
                    break
                chunk = update.text or ""
                if chunk:
                    collected.append(chunk)
                    yield chunk
        finally:
            # Ensure the underlying async stream is closed to avoid
            # "coroutine 'aclose' was never awaited" warnings and leaked
            # HTTP connections.
            aclose = getattr(aiter_, "aclose", None)
            if aclose is not None:
                try:
                    loop.run_until_complete(aclose())
                except Exception:
                    pass
            stream_aclose = getattr(stream, "aclose", None)
            if stream_aclose is not None:
                try:
                    loop.run_until_complete(stream_aclose())
                except Exception:
                    pass

    return gen(), collected


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    user_id = st.selectbox(
        "User",
        USER_OPTIONS,
        index=USER_OPTIONS.index(st.session_state.get("user_id", DEFAULT_USER_ID)),
    )
    provider = st.selectbox(
        "Model provider",
        PROVIDER_OPTIONS,
        index=PROVIDER_OPTIONS.index(
            st.session_state.get("provider", DEFAULT_PROVIDER)
        ),
    )
    # st.caption(f"Memory store: `{MEM_STORE_NAME}`")


# ---------- Top-right shutdown button ----------
_top_left, _top_right = st.columns([6, 1])
with _top_right:
    if st.button("Close", use_container_width=True):
        st.warning("Shutting down...")
        os.kill(os.getpid(), signal.SIGTERM)


# ---------- One-time init per session ----------
memory_store = get_memory_store()

provider_changed = (
    "provider" in st.session_state and st.session_state["provider"] != provider
)
user_changed = "user_id" in st.session_state and st.session_state["user_id"] != user_id

if "agent" not in st.session_state or provider_changed or user_changed:
    st.session_state["provider"] = provider
    st.session_state["user_id"] = user_id
    st.session_state["agent"] = build_agent(provider)
    st.session_state["session"] = st.session_state["agent"].create_session()
    st.session_state["messages"] = []
    if provider_changed:
        st.toast(f"Switched to {provider}. Chat reset.", icon="\U0001f501")
        st.rerun()
    if user_changed:
        st.toast(f"Switched user to {user_id}. Chat reset.", icon="\U0001f464")
        st.rerun()

# ---------- Show recalled memory ----------
with st.expander(
    f"📚 Recalled memory ({len(st.session_state['recalled'])} items)",
    expanded=False,
):
    if st.session_state["recalled"]:
        import html

        items_html = "".join(
            f"<li style='margin-bottom:6px;'>{html.escape(str(item))}</li>"
            for item in st.session_state["recalled"]
        )
        st.markdown(
            "<div style='max-height:240px; overflow-y:auto; "
            "border:1px solid var(--border-color, #888); border-radius:6px; "
            "padding:8px 12px; background-color:rgba(127,127,127,0.08); "
            "color:inherit;'>"
            f"<ol style='margin:0; padding-left:20px; color:inherit;'>"
            f"{items_html}</ol>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No prior memory found for this user.")

# ---------- Render chat history ----------
st.title("💬 Foundry Memory Chat")
st.caption("Ask me anything about Microsoft Azure.")
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Chat input ----------
prompt = st.chat_input("Ask something...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent: Agent = st.session_state["agent"]
    session = st.session_state["session"]

    with st.chat_message("assistant"):
        gen, collected = stream_agent_text(agent, prompt, session)
        st.write_stream(gen)  # type: ignore
        reply = "".join(collected)

    st.session_state["messages"].append({"role": "assistant", "content": reply})

    # Persist this turn to the memory store (best-effort).
    try:
        with st.spinner("Updating memory..."):
            memory_store.update_memory_store(
                name=MEM_STORE_NAME,
                scope=user_id,
                items=[
                    EasyInputMessageParam(content=prompt, role="user", type="message"),
                    EasyInputMessageParam(
                        content=reply, role="assistant", type="message"
                    ),
                ],
            )
    except Exception as e:
        st.warning(f"Memory update failed (turn still in chat): {e}")
