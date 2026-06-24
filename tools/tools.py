# tools/tools.py — Tool definitions + implementations for the Research Agent
#
# CONCEPT: Tool Calling
# The LLM doesn't execute tools — it outputs a structured JSON request.
# Our code reads that request, runs the real function, and feeds the result
# back to the LLM as a "tool_result" message. This loop = ReAct pattern.

import json
import math
import datetime
from config import NOTES_FILE


# ══════════════════════════════════════════════════════════════════
#  TOOL SCHEMAS  (sent to Claude so it knows what tools exist)
# ══════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression. Use this for any arithmetic, "
            "percentages, unit conversions, or numeric reasoning. "
            "Supports: +, -, *, /, **, sqrt(), log(), sin(), cos(), pi, e."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Python math expression, e.g. '15 * 847 / 100' or 'sqrt(144)'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for current information, news, facts, or research. "
            "Use this when the user asks about recent events, topics you're unsure about, "
            "or anything that needs up-to-date information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-5, default 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "save_note",
        "description": (
            "Save an important note, finding, or piece of information to a local file. "
            "Use this when the user wants to remember something, save research findings, "
            "or store any information for later."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "A short title for the note"
                },
                "content": {
                    "type": "string",
                    "description": "The content to save"
                },
                "category": {
                    "type": "string",
                    "description": "Category tag for the note (e.g. 'research', 'todo', 'idea')",
                    "default": "general"
                }
            },
            "required": ["title", "content"]
        }
    },
    {
        "name": "get_session_info",
        "description": (
            "Get information about the current session: how long it's been running, "
            "how many messages have been exchanged, current date/time, "
            "and a summary of topics discussed. Use when user asks about session stats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


# ══════════════════════════════════════════════════════════════════
#  TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════

def run_calculator(expression: str) -> str:
    """
    TOOL: Calculator
    Safely evaluates math expressions using Python's math module.
    We use a whitelist approach — only allow safe math functions.
    """
    # Safe namespace — only math functions, no builtins
    safe_globals = {
        "__builtins__": {},   # Block all Python builtins for safety
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "pow": pow,
    }

    try:
        result = eval(expression, safe_globals)
        return f"Result: {expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as ex:
        return f"Error evaluating '{expression}': {str(ex)}"


def run_web_search(query: str, num_results: int = 3) -> str:
    """
    TOOL: Web Search (Simulated)
    In a real deployment, connect to SerpAPI, Tavily, or Brave Search.
    Here we simulate results so the agent pipeline works end-to-end
    without requiring an extra API key.
    """
    # ── Simulated search results (replace with real API in production) ──
    simulated_results = {
        "python": [
            {"title": "Python 3.13 Released", "url": "https://python.org/news", 
             "snippet": "Python 3.13 brings improved error messages, a new REPL, and experimental free-threaded mode (no GIL)."},
            {"title": "Top Python Libraries 2025", "url": "https://realpython.com",
             "snippet": "Pydantic v2, Polars, and uv (fast package manager) are dominating Python workflows in 2025."},
            {"title": "Python for AI/ML", "url": "https://towardsdatascience.com",
             "snippet": "Python remains the #1 language for ML, with LangChain, LlamaIndex, and DSPy leading the agent framework space."}
        ],
        "ai": [
            {"title": "Large Language Models Overview", "url": "https://arxiv.org",
             "snippet": "LLMs like GPT-4, Claude, and Gemini use transformer architecture trained on massive datasets via next-token prediction."},
            {"title": "AI Agents in 2025", "url": "https://huggingface.co/blog",
             "snippet": "Agentic AI systems now handle multi-step reasoning, tool use, and memory management autonomously."},
            {"title": "Retrieval-Augmented Generation (RAG)", "url": "https://langchain.com",
             "snippet": "RAG combines vector databases with LLMs to give models access to external, up-to-date knowledge."}
        ],
        "machine learning": [
            {"title": "What is Machine Learning?", "url": "https://scikit-learn.org",
             "snippet": "ML teaches computers to learn from data. Key paradigms: supervised, unsupervised, and reinforcement learning."},
            {"title": "Deep Learning Advances", "url": "https://deepmind.google",
             "snippet": "Transformer models now dominate not just NLP but also vision, audio, and multi-modal tasks."},
        ]
    }

    # Find best matching simulated results
    query_lower = query.lower()
    matched_results = []
    for keyword, results in simulated_results.items():
        if keyword in query_lower:
            matched_results = results[:num_results]
            break

    # Default results if nothing matched
    if not matched_results:
        matched_results = [
            {"title": f"Search results for: {query}", "url": "https://google.com",
             "snippet": f"Found information about '{query}'. This is a simulated result — connect a real search API (Tavily, SerpAPI) for live data."},
            {"title": "Wikipedia Overview", "url": f"https://wikipedia.org/wiki/{query.replace(' ', '_')}",
             "snippet": f"Wikipedia provides a comprehensive overview of {query} including history, applications, and current developments."},
        ]

    # Format results
    output = f"🔍 Search results for: '{query}'\n\n"
    for i, r in enumerate(matched_results[:num_results], 1):
        output += f"{i}. **{r['title']}**\n"
        output += f"   {r['snippet']}\n"
        output += f"   Source: {r['url']}\n\n"

    output += "⚠️  Note: These are simulated results. To get live data, integrate Tavily or SerpAPI."
    return output


def run_save_note(title: str, content: str, category: str = "general") -> str:
    """
    TOOL: Save Note
    Persists a note to a local text file with timestamp.
    This shows how agents can have side effects (write to disk, DB, etc.)
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    note_entry = f"""
{'='*50}
📝 [{category.upper()}] {title}
🕐 Saved at: {timestamp}
{'='*50}
{content}

"""
    try:
        with open(NOTES_FILE, "a", encoding="utf-8") as f:
            f.write(note_entry)
        return f"✅ Note '{title}' saved successfully to {NOTES_FILE} at {timestamp}."
    except Exception as ex:
        return f"❌ Failed to save note: {str(ex)}"


# Session tracker (shared state for get_session_info)
_session_start = datetime.datetime.now()
_message_count = 0

def increment_message_count():
    global _message_count
    _message_count += 1

def run_get_session_info() -> str:
    """
    TOOL: Session Info
    Returns metadata about the current agent session.
    Demonstrates that tools can read internal agent state.
    """
    now = datetime.datetime.now()
    elapsed = now - _session_start
    minutes = int(elapsed.total_seconds() // 60)
    seconds = int(elapsed.total_seconds() % 60)

    return (
        f"📊 Session Information:\n"
        f"  • Started: {_session_start.strftime('%H:%M:%S')}\n"
        f"  • Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  • Duration: {minutes}m {seconds}s\n"
        f"  • Messages exchanged: {_message_count}\n"
        f"  • Notes file: {NOTES_FILE}"
    )


# ══════════════════════════════════════════════════════════════════
#  TOOL DISPATCHER
# ══════════════════════════════════════════════════════════════════

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Central dispatcher: routes a tool call from the LLM to the correct function.
    This is called in the agent loop after the LLM requests a tool.
    """
    if tool_name == "calculator":
        return run_calculator(tool_input["expression"])

    elif tool_name == "web_search":
        return run_web_search(
            query=tool_input["query"],
            num_results=tool_input.get("num_results", 3)
        )

    elif tool_name == "save_note":
        return run_save_note(
            title=tool_input["title"],
            content=tool_input["content"],
            category=tool_input.get("category", "general")
        )

    elif tool_name == "get_session_info":
        return run_get_session_info()

    else:
        return f"❌ Unknown tool: '{tool_name}'. Available tools: calculator, web_search, save_note, get_session_info."
