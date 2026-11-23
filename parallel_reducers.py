
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.1
)

# Merge scores returned by different nodes
def merge_score_dicts(existing: dict, newupdate: dict) -> dict:
    if existing is None:
        existing = {}

    return {**existing, **newupdate}


# Shared state
class AnalyzerState(TypedDict):
    raw_text: str
    safety_score: Annotated[dict[str, int], merge_score_dicts]


# Helper function to get a score from the LLM
def get_score(prompt: str) -> int:
    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
        return max(0, min(100, score))
    except (ValueError, TypeError):
        return 0


# Node 1: Toxicity analysis
def toxicity_node(state: AnalyzerState) -> dict:
    print("\n[Branch 1] Analyzing toxicity...")

    prompt = f"""
Analyze the following text for profanity, aggression,
hate speech, and toxicity.

Give a score from 0 to 100:
0 = perfectly clean
100 = highly toxic

Return only an integer.

Text:
{state["raw_text"]}
"""

    score = get_score(prompt)

    return {
        "safety_score": {"toxicity_level": score}
    }


# Node 2: Copyright risk analysis
def copyright_node(state: AnalyzerState) -> dict:
    print("\n[Branch 2] Analyzing copyright risk...")

    prompt = f"""
Assess the text for potential copyright or plagiarism risk.
Do not assume plagiarism solely from common phrases.

Give a score from 0 to 100:
0 = low apparent risk
100 = high apparent risk

Return only an integer.

Text:
{state["raw_text"]}
"""

    score = get_score(prompt)

    return {
        "safety_score": {"copyright_level": score}
    }


# Node 3: Cultural sensitivity analysis
def culture_node(state: AnalyzerState) -> dict:
    print("\n[Branch 3] Analyzing cultural sensitivity...")

    prompt = f"""
Analyze the text for cultural insensitivity,
regional stereotypes, and potentially offensive language.

Give a score from 0 to 100:
0 = no apparent concern
100 = highly offensive

Return only an integer.

Text:
{state["raw_text"]}
"""

    score = get_score(prompt)

    return {
        "safety_score": {"cultural_level": score}
    }


# Build the graph
builder = StateGraph(AnalyzerState)

builder.add_node("toxicity_node", toxicity_node)
builder.add_node("copyright_node", copyright_node)
builder.add_node("culture_node", culture_node)


# Run all three nodes in parallel
builder.add_edge(START, "toxicity_node")
builder.add_edge(START, "copyright_node")
builder.add_edge(START, "culture_node")

builder.add_edge("toxicity_node", END)
builder.add_edge("copyright_node", END)
builder.add_edge("culture_node", END)


# Compile the graph
app = builder.compile()


# Input
sample_script = """
Welcome back to the stream! Today I will show you
a script copied directly from an online tutorial.
Some of the language is insulting and offensive.
"""

initial_state = {
    "raw_text": sample_script,
    "safety_score": {}
}

# Execute the graph
final_state = app.invoke(initial_state)

print("\nFinal safety scores:")
print(final_state["safety_score"])
