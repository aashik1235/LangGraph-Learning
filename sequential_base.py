
from typing import TypedDict

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END


# 1. Load environment variables
load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.6
)


# 2. Define the shared state
class PipelineState(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str


# 3. Stage 1: Edit the text
def editor_node(state: PipelineState) -> dict:
    """Fix grammar, spelling, and readability."""

    print("\n--- [Stage 1] Executing the editor node ---")

    prompt = (
        "You are an expert copyeditor. Clean up the following raw text. "
        "Correct grammatical errors and spelling mistakes, and improve "
        "the flow and readability while preserving the original meaning. "
        "Return only the edited text, without explanations.\n\n"
        f"Raw text:\n{state['raw_input']}"
    )

    response = llm.invoke(prompt)

    return {
        "edited_text": response.content.strip()
    }


# 4. Stage 2: Write the video script
def scriptwriter_node(state: PipelineState) -> dict:
    """Transform edited text into an engaging video script."""

    print("\n--- [Stage 2] Executing the scriptwriter node ---")

    prompt = (
        "You are a charismatic YouTube content creator. "
        "Transform the following edited text into an engaging, "
        "punchy, conversational video script. Start with a powerful "
        "hook. Make it sound natural, lively, relatable, and easy to follow. "
        "Return only the script, without explanations.\n\n"
        f"Edited text:\n{state['edited_text']}"
    )

    response = llm.invoke(prompt)

    return {
        "script_text": response.content.strip()
    }


# 5. Stage 3: Translate into Hinglish
def translator_node(state: PipelineState) -> dict:
    """Translate the script into natural Hinglish."""

    print("\n--- [Stage 3] Executing the Hinglish translator node ---")

    prompt = (
        "You are an expert content localizer for the Nepali market. "
        "Transform the following video script into natural, fluent Hinglish. "
        "Do not translate word-for-word. Adapt expressions and phrasing "
        "for a Nepali audience. Alternate naturally between Hindi and English. "
        "Keep the script conversational and easy to understand. "
        "Preserve the original meaning and engaging tone. "
        "Return only the translated script, without explanations.\n\n"
        f"Video script:\n{state['script_text']}"
    )

    response = llm.invoke(prompt)

    return {
        "final_output": response.content.strip()
    }


# 6. Create the graph
graph = StateGraph(PipelineState)


# 7. Add nodes
graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)


# 8. Connect nodes sequentially
graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)


# 9. Compile the graph
app = graph.compile()


# 10. Run the workflow
result = app.invoke({
    "raw_input": (
        "AI agents are the future of technology. They can think, "
        "plan, and act on their own. LangGraph helps you build "
        "these agents with proper control and memory."
    )
})


# 11. Display the final output
print("\nYour final Hinglish script:\n")
print(result["final_output"])
