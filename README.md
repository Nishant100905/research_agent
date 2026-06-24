# 🤖 Personal Research Assistant Agent

A fully functional AI agent built in Python that demonstrates core LLM engineering concepts.

## Concepts Covered

| Concept | Implementation |
|---|---|
| **LLM Basics** | Google Gemini API (via OpenAI compat) — system prompts, message formatting, token usage |
| **Tool Calling** | 4 tools: web search, calculator, note saver, session timer |
| **Short-term Memory** | Rolling conversation history passed on every API call |
| **Memory Management** | Auto-summarization when history exceeds token threshold |

## Project Structure

```
research_agent/
├── agent/
│   └── agent.py          # Core agent loop (ReAct: Reason → Act → Observe)
├── memory/
│   └── memory_manager.py # Short-term memory + summarization
├── tools/
│   └── tools.py          # Tool definitions and implementations
├── utils/
│   └── logger.py         # Color-coded terminal logger
├── main.py               # Entry point (CLI chat interface)
├── config.py             # Configuration (model, token limits)
└── requirements.txt
```

## Setup

```bash
pip install openai
export GEMINI_API_KEY=your_key_here
python main.py
```

## How It Works

1. User sends a message
2. Agent adds it to short-term memory (conversation history)
3. LLM decides whether to call a tool or respond directly
4. If tool called → result added back to history → LLM reasons again
5. Memory manager checks token count — if too long, summarizes older turns
6. Final response returned to user

## Example Interactions

```
You: What is 15% of 847?
Agent: [calls calculator] → 15% of 847 is 127.05

You: Search for latest Python features
Agent: [calls web_search] → Here's what I found...

You: Save a note: Remember to review transformers paper
Agent: [calls save_note] → Note saved!

You: What have we talked about so far?
Agent: [reads from short-term memory] → Here's our conversation summary...
```
