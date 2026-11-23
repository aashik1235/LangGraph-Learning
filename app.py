import os
import sys
import types
from pathlib import Path

import streamlit as st


# STEP 1: Configuration

BACKEND_FILE = "conditional_RAG.py"
CLI_MARKER = "# STEP 7"

PROGRAMMES = ["BCA", "BBA", "B.Com (H)"]

st.set_page_config(
    page_title="Campus Assistant",
    page_icon="🎓"
)

st.title("🎓 Campus Assistant")
st.write("Ask questions about academics, fees, and college rules.")


# STEP 2: Load the LangGraph application

@st.cache_resource(show_spinner="Loading college documents...")
def load_graph():

    backend_path = Path(__file__).resolve().parent / BACKEND_FILE

    if not backend_path.exists():
        raise FileNotFoundError(f"{BACKEND_FILE} not found.")

    source = backend_path.read_text(encoding="utf-8")

    if CLI_MARKER not in source:
        raise ValueError(f"{CLI_MARKER} not found in backend file.")

    # Remove the console input loop
    source = source.split(CLI_MARKER)[0]

    # Set the working directory for PDF paths
    os.chdir(backend_path.parent)

    module = types.ModuleType("backend_graph")
    module.__file__ = str(backend_path)
    sys.modules["backend_graph"] = module

    exec(compile(source, str(backend_path), "exec"), module.__dict__)

    return module.app


try:
    graph_app = load_graph()
except Exception as e:
    st.error(f"Could not load assistant: {e}")
    st.stop()


# STEP 3: Sidebar settings

programme = st.sidebar.selectbox(
    "Choose your programme",
    PROGRAMMES
)

show_context = st.sidebar.checkbox(
    "Show retrieved passages",
    value=True
)

st.sidebar.subheader("Example questions")

examples = [
    "What is the minimum attendance requirement?",
    "What is the tuition fee per semester?",
    "How does the grading system work?",
    "What is the refund policy?",
    "Hi, how are you?"
]

for example in examples:
    if st.sidebar.button(example, use_container_width=True):
        st.session_state["pending_query"] = example

if st.sidebar.button("Clear chat", use_container_width=True):
    st.session_state["messages"] = []
    st.rerun()


# STEP 4: Initialize chat history

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# STEP 5: Display previous messages

for message in st.session_state["messages"]:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        context = message.get("context")

        if (
            message["role"] == "assistant"
            and show_context
            and context
            and context != "NO_RETRIEVAL_NEEDED"
        ):
            with st.expander("View retrieved passages"):
                st.text(context)


# STEP 6: Get the user's question

query = st.chat_input("Ask your question...")

if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")


# STEP 7: Process the question

if query:

    st.session_state["messages"].append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                result = graph_app.invoke({
                    "programme": programme,
                    "messages": [("human", query)]
                })

                answer = result["messages"][-1].content

                entry = {
                    "role": "assistant",
                    "content": answer,
                    "context": result.get("retrieved_context")
                }

            except Exception as e:
                entry = {
                    "role": "assistant",
                    "content": f"Something went wrong: {e}"
                }

        st.markdown(entry["content"])

        context = entry.get("context")

        if (
            show_context
            and context
            and context != "NO_RETRIEVAL_NEEDED"
        ):
            with st.expander("View retrieved passages"):
                st.text(context)

    st.session_state["messages"].append(entry)      