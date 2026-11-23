from typing import TypedDict, Annotated, Literal

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()


# STEP 1: Build the RAG retrievers

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_retriever(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )


academic_retriever = build_retriever(
    "cond_data/academics_handbook.pdf"
)

fee_retriever = build_retriever(
    "cond_data/fee_structure.pdf"
)


# STEP 2: Initialize the LLM

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.4
)


# STEP 3: Define the LangGraph state

class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# STEP 4: Create the graph nodes

def classifier_node(state: State) -> dict:
    last_message = state["messages"][-1].content

    prompt = (
        "Classify the student query into exactly one category: "
        "academic, fee, or general.\n\n"
        "Use academic for questions about attendance, exams, "
        "grading, credits, promotion, course structure, "
        "summer training, or degree requirements.\n"
        "Use fee for questions about tuition payments, refunds, "
        "late charges, or scholarships.\n"
        "Use general for greetings, casual conversations, "
        "or questions unrelated to academic rules and fees.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if category.startswith("academic"):
        category = "academic"
    elif category.startswith("fee"):
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}


def academic_rag_node(state: State) -> dict:
    query = state["messages"][-1].content
    docs = academic_retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED"
    }


def response_node(state: State) -> dict:
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant helping a "
            f"{programme} student.\n"
            "Answer the following general query naturally "
            "using your general knowledge.\n\n"
            f"Question: {query}"
        )

    else:
        prompt = (
            f"You are a friendly college assistant helping a "
            f"{programme} student.\n\n"
            "Use the following context from the official college "
            "documents to answer the question accurately.\n"
            f"Student programme: {programme}\n\n"
            "If the context contains information for different "
            "programmes, prioritize the information relevant to "
            f"{programme} when available.\n\n"
            "Do not invent college rules, fees, or figures. "
            "If the context does not contain enough information, "
            "clearly say that you could not find the answer "
            "in the retrieved documents.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)

    return {
        "messages": [
            ("assistant", response.content.strip())
        ]
    }


# STEP 5: Define the routing function

def route_query(
    state: State
) -> Literal["academic_rag", "fee_rag", "general"]:

    if state["query_type"] == "academic":
        return "academic_rag"

    elif state["query_type"] == "fee":
        return "fee_rag"

    else:
        return "general"


# STEP 6: Build the LangGraph workflow

graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

graph.add_edge(START, "classifier")

graph.add_conditional_edges(
    "classifier",
    route_query
)

graph.add_edge("academic_rag", "response")
graph.add_edge("fee_rag", "response")
graph.add_edge("general", "response")

graph.add_edge("response", END)

app = graph.compile()


# STEP 7: Run the college assistant

print("Welcome to the College Assistant!")
print("Which programme are you enrolled in?")
print("1. BCA")
print("2. BBA")
print("3. B.Com (H)")

choice = input("Enter 1, 2, or 3: ").strip()

programme_map = {
    "1": "BCA",
    "2": "BBA",
    "3": "B.Com (H)"
}

student_programme = programme_map.get(choice, "BCA")

print(f"\nGreat! You are set as a {student_programme} student.")
print("Ask your questions. Type 'exit' or 'quit' to stop.\n")

while True:
    user_query = input("You: ").strip()

    if user_query.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not user_query:
        continue

    result = app.invoke({
        "programme": student_programme,
        "messages": [
            ("human", user_query)
        ]
    })

    print("\nAssistant:", result["messages"][-1].content)
    print()