```python
import streamlit as st
import sqlite3
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent


# =========================================================
# 1. STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("AI Research Assistant.")


# =========================================================
# 2. SQLITE CHECKPOINTER
# =========================================================

conn = sqlite3.connect(
    "test.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)


# =========================================================
# 3. GROQ MODEL
# =========================================================

# IMPORTANT:
# Put your API key in .streamlit/secrets.toml
#
# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

model = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY
)


# =========================================================
# 4. SEARCH TOOL
# =========================================================

search = DuckDuckGoSearchRun()


@tool
def search_tool(query: str):
    """
    Search the web for up-to-date information.
    Use this tool when the user asks for current or
    web-based information.
    """

    response = search.invoke(query)

    return response


# =========================================================
# 5. SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
Role:
You are a helpful, knowledgeable, and reliable AI assistant.

Task:
Answer the user's questions clearly, accurately, and concisely.
If the question requires explanation, break the answer into
simple, easy-to-understand steps.

Output Format:

- Use paragraphs for explanations.
- Use bullet points for key points, steps, and lists.
- Use tables when comparing multiple items or presenting
  structured information.
- Use code blocks when providing programming code.
- Use examples whenever they help clarify the concept.

Guidelines:

- Understand the user's intent before answering.
- Give accurate and relevant information.
- Avoid unnecessary repetition.
- For technical topics, explain concepts from basic to advanced
  when appropriate.
- If the question is ambiguous, ask for clarification instead
  of making assumptions.
- Use the search tool when current or web-based information
  is required.
- Keep the response well-structured and easy to read.
"""


# =========================================================
# 6. LANGGRAPH AGENT
# =========================================================

agent = create_react_agent(
    model=model,
    tools=[search_tool],
    checkpointer=checkpointer
)


# =========================================================
# 7. SESSION STATE
# =========================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# =========================================================
# 8. THREAD FUNCTIONS
# =========================================================

def get_thread_ids():
    """
    Get all unique thread IDs stored in SQLite.
    """

    thread_ids = set()

    try:
        for checkpoint in checkpointer.list(None):

            config = checkpoint.config

            if config and "configurable" in config:

                thread_id = config["configurable"].get(
                    "thread_id"
                )

                if thread_id:
                    thread_ids.add(thread_id)

    except Exception:
        pass

    return list(thread_ids)


def get_thread_title(thread_id):
    """
    Get the first user message of a thread and use it
    as the thread title.
    """

    try:

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        state = agent.get_state(config)

        messages = state.values.get("messages", [])

        for message in messages:

            if getattr(message, "type", "") == "human":

                content = message.content

                if isinstance(content, str):

                    content = content.strip()

                    if len(content) > 35:
                        return content[:35] + "..."

                    return content

    except Exception:
        pass

    return "New Conversation"


# =========================================================
# 9. SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🧵 Threads")

    # New Chat button
    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.thread_id = str(uuid.uuid4())

        st.rerun()

    st.divider()

    thread_ids = get_thread_ids()

    # Sort threads
    thread_ids = list(thread_ids)

    if thread_ids:

        for thread_id in thread_ids:

            title = get_thread_title(thread_id)

            is_current = (
                thread_id ==
                st.session_state.thread_id
            )

            if is_current:
                button_label = f"🟢 {title}"
            else:
                button_label = f"💬 {title}"

            if st.button(
                button_label,
                key=f"thread_{thread_id}",
                use_container_width=True
            ):

                st.session_state.thread_id = thread_id

                st.rerun()

    else:

        st.info(
            "No previous conversations yet."
        )

    st.divider()

    st.caption(
        f"Current Thread:\n"
        f"`{st.session_state.thread_id}`"
    )


# =========================================================
# 10. CURRENT THREAD CONFIG
# =========================================================

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# =========================================================
# 11. LOAD CURRENT CONVERSATION
# =========================================================

try:

    state = agent.get_state(config)

    messages = state.values.get(
        "messages",
        []
    )

except Exception:

    messages = []


# =========================================================
# 12. DISPLAY CHAT HISTORY
# =========================================================

for message in messages:

    message_type = getattr(
        message,
        "type",
        ""
    )

    content = getattr(
        message,
        "content",
        ""
    )

    # User message
    if message_type == "human":

        with st.chat_message("user"):
            st.markdown(content)

    # AI message
    elif message_type == "ai":

        # Sometimes AI messages may contain
        # tool calls but no text.
        if content:

            with st.chat_message("assistant"):
                st.markdown(content)


# =========================================================
# 13. USER INPUT
# =========================================================

question = st.chat_input(
    "Ask me anything..."
)


if question:

    # Display user question immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Run LangGraph agent
    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            try:

                result = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT
                            },
                            {
                                "role": "user",
                                "content": question
                            }
                        ]
                    },
                    config=config
                )

                # Get final AI response
                final_message = result[
                    "messages"
                ][-1]

                answer = final_message.content

                st.markdown(answer)

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )
```

### 1. Put your Groq API key in `secrets.toml`

**Do not put the API key directly inside `app.py`.** The key in your original code is exposed, so I recommend **revoking/rotating it** and creating a new one.

Create:

```text
.streamlit/
    secrets.toml
```

Inside:

```toml
GROQ_API_KEY = "YOUR_NEW_GROQ_API_KEY"
```

Then your Python code safely uses:

```python
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
```

### 2. Install the packages

```bash
pip install streamlit langgraph langgraph-checkpoint-sqlite langchain-groq langchain-community duckduckgo-search
```

Depending on your installed versions, you may also need:

```bash
pip install langchain-core
```

### 3. Run the application

```bash
streamlit run app.py
```

You should get something like:

```text
 AI Research Assistant

```

### How the thread system works

The important part is:

```python
config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}
```

Suppose the user starts:

```text
Thread 1
"What is LangGraph?"
```

LangGraph stores the conversation in:

```text
test.db
```

If the user then clicks:

```text
 New Chat
```

a new UUID is generated:

```python
st.session_state.thread_id = str(uuid.uuid4())
```

Now they have a completely separate conversation.

Later, when they click the old thread in the sidebar, the same:

```python
thread_id
```

is passed back to LangGraph, so the previous conversation can be restored.

### One important correction in your original code

You had:

```python
@tool
def search_tool(query: str):

    response = search.invoke(prompt)

    return response
```

This means the users actual:

```python
query
```

is **never sent to DuckDuckGo**.

It should be:

```python
@tool
def search_tool(query: str):

    response = search.invoke(query)

    return response
```

That is an important fix.

### One more improvement I'd recommend

For a **resume-level project**, I would take this one step further and make the sidebar look like:

```text
 Conversations

 New Chat

Today
   What is LangGraph?
   Explain RAG architecture

Yesterday
   FastAPI deployment
   Machine Learning interview


⚙️ Settings
🗑️ Clear conversations
```

And add **streaming responses**, **thread deletion**, **search/tool status**, and **source citations**. That would make this a much stronger **LangGraph + Streamlit + Agentic AI project for your resume**.
