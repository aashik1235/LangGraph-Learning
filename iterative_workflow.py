
import time
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv

load_dotenv()


# Tools

search_tool = TavilySearch(max_results=1)
tools = [search_tool]


# LLM

writer_llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7
)

writer_llm_with_tools = writer_llm.bind_tools(tools)


# Reviewer LLM

reviewer_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)


# State

class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


# Writer prompt

WRITER_SYSTEM_PROMPT = """
You are an expert LinkedIn content writer.

1. Start with a strong hook.
2. Focus on one clear message.
3. Use simple professional English.
4. Keep the post around 150-200 words.
5. Use short paragraphs and bullets when useful.
6. Include valuable insights and practical advice.
7. Use web search for current information when needed.
8. Never invent facts or statistics.
9. End with an engaging question or CTA.
10. Do not use hashtags or unnecessary emojis.

If previous feedback is provided, fix every issue.

Return only the final LinkedIn post.
"""


# Writer node

def writer_node(state: State) -> dict:

    history = state.get("messages", [])
    attempt = state.get("attempt", 0)

    # If last message is from Tavily,
    # continue the same writing attempt.
    if history and isinstance(history[-1], ToolMessage):
        user_message = None

    else:
        attempt += 1

        topic = state["topic"]
        previous_feedback = state.get("review_feedback", "")

        if attempt == 1:
            user_message = (
                f"Write a LinkedIn post on this topic: {topic}. "
                f"If you need current information, search the web first."
            )

        else:
            user_message = (
                f"Your previous draft on '{topic}' was rejected.\n\n"
                f"Reviewer's feedback:\n{previous_feedback}\n\n"
                f"Write a new improved draft fixing every issue."
            )

    # Create messages for Gemini
    messages = [("system", WRITER_SYSTEM_PROMPT)] + history

    if user_message:
        messages.append(("human", user_message))

    # Measure Gemini execution time
    print("\nGemini is generating the post...")

    start_time = time.perf_counter()

    response = writer_llm_with_tools.invoke(messages)

    gemini_time = time.perf_counter() - start_time

    print(f"Gemini finished in {gemini_time:.2f} seconds.")

    new_messages = []

    if user_message:
        new_messages.append(("human", user_message))

    new_messages.append(response)

    return {
        "messages": new_messages,
        "attempt": attempt
    }


# Tool node

tool_node = ToolNode(tools)


# Extract draft node

def extract_draft_node(state: State) -> dict:

    last_message = state["messages"][-1]

    draft = last_message.content

    # Convert list of text blocks into a normal string
    if isinstance(draft, list):
        draft = "".join(
            block["text"]
            for block in draft
            if block.get("type") == "text"
        )

    print(f"\n\nGenerated post:\n{draft}\n")

    return {
        "draft": draft
    }


# Reviewer prompt

REVIEWER_SYSTEM_PROMPT = """
You are a strict but fair LinkedIn content reviewer.

Evaluate the post based on:

1. Strong hook.
2. One clear takeaway.
3. Easy readability.
4. Approximately 150-200 words.
5. Engaging question or CTA at the end.
6. Professional but natural tone.
7. No hashtags.

Approve only if all criteria are met.
Reject if an important criterion is missing.

Respond exactly in this format:

Verdict: APPROVED or REJECTED

Feedback: One short paragraph explaining why.

Do not include anything else.
"""


# Reviewer node

def reviewer_node(state: State) -> dict:

    # TEMPORARY TEST: Reject the first draft
    if state["attempt"] == 1:

        feedback = (
            "The post needs a stronger opening hook. "
            "Add a more practical example and make the "
            "main takeaway clearer."
        )

        print("\n[Verdict: REJECTED]")
        print(f"[Feedback: {feedback}]")

        return {
            "review_feedback": feedback,
            "is_approved": False
        }

    # Actual Groq reviewer
    draft = state["draft"]

    prompt = (
        f"Review this LinkedIn post:\n\n"
        f"{draft}\n\n"
        f"Give your review."
    )

    # Measure Groq execution time
    print("\nGroq is reviewing the post...")

    start_time = time.perf_counter()

    response = reviewer_llm.invoke(
        [
            ("system", REVIEWER_SYSTEM_PROMPT),
            ("human", prompt)
        ]
    )

    groq_time = time.perf_counter() - start_time

    print(f"Groq finished in {groq_time:.2f} seconds.")

    review_text = response.content.strip()

    # Extract verdict
    verdict_line = next(
        (
            line.strip()
            for line in review_text.splitlines()
            if line.strip().lower().startswith("verdict:")
        ),
        ""
    )

    is_approved = verdict_line.upper() == "VERDICT: APPROVED"

    # Extract feedback
    if "Feedback:" in review_text:
        feedback = review_text.split("Feedback:", 1)[1].strip()
    else:
        feedback = review_text

    verdict = "APPROVED" if is_approved else "REJECTED"

    print(f"\n[Verdict: {verdict}]")
    print(f"[Feedback: {feedback}]")

    return {
        "review_feedback": feedback,
        "is_approved": is_approved
    }


# Router functions

def should_use_tool(state: State):

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        print("\nTavily search requested.")
        return "tools"

    return "extract_draft"


def should_stop_looping(state: State):

    if state["is_approved"]:
        print("\nPost has been approved!")
        return END

    if state["attempt"] >= 3:
        print("\nReached maximum attempts.")
        return END

    return "writer"


# Build graph

graph = StateGraph(State)

graph.add_node("writer", writer_node)
graph.add_node("tools", tool_node)
graph.add_node("extract_draft", extract_draft_node)
graph.add_node("reviewer", reviewer_node)

graph.add_edge(START, "writer")

graph.add_conditional_edges(
    "writer",
    should_use_tool
)

# After Tavily search, return to writer
graph.add_edge("tools", "writer")

graph.add_edge("extract_draft", "reviewer")

graph.add_conditional_edges(
    "reviewer",
    should_stop_looping
)


# Compile graph

app = graph.compile()


# Main program

print("=" * 50)
print("Welcome to the LinkedIn Post Generator")
print("=" * 50)

print("\nThis tool will draft, review, and improve your post.")

topic = input("\nWhat topic do you want a LinkedIn post about?\n").strip()

if not topic:

    print("Please enter a valid topic.")

else:

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0
    }

    # Measure complete workflow time
    workflow_start = time.perf_counter()

    final_state = app.invoke(initial_state)

    total_time = time.perf_counter() - workflow_start

    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)

    print("\nTopic:", final_state["topic"])
    print("Attempts:", final_state["attempt"])
    print("Approved:", final_state["is_approved"])

    print("\nFinal LinkedIn Post:")
    print(final_state["draft"])

    print("\nReviewer Feedback:")
    print(final_state["review_feedback"])

    print(f"\nTotal workflow time: {total_time:.2f} seconds")
