# memory/memory_manager.py — Short-term Memory + Memory Management
#
# CONCEPT: Short-term Memory
# Every LLM call is stateless — it has no memory of previous calls.
# We simulate memory by maintaining a list of messages and sending
# the full history on every API call. This is "in-context memory".
#
# CONCEPT: Memory Management
# The context window is limited (~200k tokens for Claude, but we budget less).
# When conversation history grows too long, we summarize older turns using
# the LLM itself, then replace them with a compact summary. This keeps
# the agent functional indefinitely without losing important context.

import openai
from config import (
    MODEL, MAX_TOKENS,
    SHORT_TERM_MEMORY_LIMIT,
    SUMMARY_KEEP_RECENT
)
from utils.logger import log_memory_event, log_memory_status


class MemoryManager:
    """
    Manages the agent's conversational memory.
    
    Architecture:
    ┌─────────────────────────────────────────────┐
    │              Context Window                 │
    │  ┌──────────────┐  ┌─────────────────────┐ │
    │  │    Summary   │  │   Recent Turns      │ │
    │  │  (condensed  │  │  (full detail,      │ │
    │  │  old turns)  │  │   last N turns)     │ │
    │  └──────────────┘  └─────────────────────┘ │
    └─────────────────────────────────────────────┘
    """

    def __init__(self):
        self.messages: list[dict] = []       # Full conversation history
        self.summary: str | None = None      # Summary of older turns (if any)
        self.total_turns: int = 0            # Lifetime turn counter
        self.summarizations_done: int = 0   # How many times we've summarized

    # ── Public API ──────────────────────────────────────────────

    def add_user_message(self, content: str):
        """Add a user message to short-term memory."""
        self.messages.append({"role": "user", "content": content})
        self.total_turns += 1

    def add_assistant_message(self, message):
        """Add an assistant message (str, dict, or ChatCompletionMessage)."""
        if hasattr(message, "model_dump"):
            self.messages.append(message.model_dump(exclude_none=True))
        elif isinstance(message, dict):
            self.messages.append(message)
        else:
            self.messages.append({"role": "assistant", "content": message})

    def add_tool_result(self, tool_use_id: str, result: str):
        """
        Add a tool result to memory.
        Tool results must be wrapped in the OpenAI-specific format.
        """
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_use_id,
            "content": result
        })

    def get_messages(self) -> list[dict]:
        """
        Return messages to send to the API.
        If a summary exists, prepend it as a system-like user message.
        """
        if self.summary:
            summary_msg = {
                "role": "user",
                "content": f"[CONVERSATION SUMMARY — earlier context]\n{self.summary}"
            }
            summary_ack = {
                "role": "assistant",
                "content": "I have the summary of our earlier conversation. I'll use it as context going forward."
            }
            return [summary_msg, summary_ack] + self.messages
        return self.messages

    def check_and_compress(self, client: openai.OpenAI):
        """
        Check if memory needs compression. If conversation history
        is too long, summarize older turns to free up context space.
        
        Called after every agent turn.
        """
        turn_count = len(self.messages)
        log_memory_status(
            turn_count=turn_count,
            limit=SHORT_TERM_MEMORY_LIMIT * 2,  # *2 because each turn = 2 messages
            summarized=self.summary is not None
        )

        # Trigger summarization when message count exceeds limit
        if turn_count >= SHORT_TERM_MEMORY_LIMIT * 2:
            log_memory_event(
                f"Memory limit reached ({turn_count} messages). "
                f"Summarizing older turns..."
            )
            self._summarize_old_turns(client)

    def get_stats(self) -> dict:
        """Return memory statistics for display."""
        return {
            "current_messages": len(self.messages),
            "total_turns": self.total_turns,
            "has_summary": self.summary is not None,
            "summarizations": self.summarizations_done,
            "summary_preview": self.summary[:100] + "..." if self.summary else None
        }

    def clear(self):
        """Reset all memory (new session)."""
        self.messages = []
        self.summary = None
        self.total_turns = 0
        log_memory_event("Memory cleared. Starting fresh session.")

    # ── Private Methods ─────────────────────────────────────────

    def _summarize_old_turns(self, client: openai.OpenAI):
        """
        Summarize older conversation turns using the LLM.
        
        Strategy:
        1. Take all messages EXCEPT the last SUMMARY_KEEP_RECENT turns
        2. Ask the LLM to produce a concise summary
        3. Replace old messages with the summary
        4. Keep the most recent turns intact for full detail
        """
        # Split: old turns to summarize vs recent turns to keep
        keep_count = SUMMARY_KEEP_RECENT * 2   # *2 for user+assistant pairs
        if len(self.messages) <= keep_count:
            return  # Not enough messages to split

        old_turns = self.messages[:-keep_count]
        recent_turns = self.messages[-keep_count:]

        # Format old turns for summarization
        conversation_text = self._format_turns_for_summary(old_turns)

        # Include existing summary context if we have one
        existing_summary_context = ""
        if self.summary:
            existing_summary_context = f"\nExisting earlier summary:\n{self.summary}\n\nNew turns to add:"

        # Call LLM to create summary
        summary_prompt = f"""You are summarizing a conversation between a user and an AI research assistant.
Create a concise but complete summary that captures:
- Main topics discussed
- Key facts, answers, or findings
- Any notes saved or tasks completed
- Important context needed to continue the conversation

{existing_summary_context}

Conversation turns to summarize:
{conversation_text}

Provide a clear, structured summary (3-8 sentences or bullet points):"""

        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": summary_prompt}]
            )
            new_summary = response.choices[0].message.content

            # If we had a previous summary, merge it
            if self.summary:
                self.summary = f"{new_summary}"
            else:
                self.summary = new_summary

            # Replace old messages with recent ones only
            self.messages = recent_turns
            self.summarizations_done += 1

            log_memory_event(
                f"Summarization #{self.summarizations_done} complete. "
                f"Compressed {len(old_turns)} messages → summary. "
                f"Keeping {len(recent_turns)} recent messages."
            )
            log_memory_event(f"Summary preview: {self.summary[:100]}...")

        except Exception as ex:
            log_memory_event(f"Summarization failed: {ex}. Keeping all messages.")

    def _format_turns_for_summary(self, messages: list[dict]) -> str:
        """Convert message dicts to readable text for summarization."""
        lines = []
        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content")

            if isinstance(content, str) and content:
                lines.append(f"{role}: {content}")
            
            # Handle tool calls by the assistant
            if msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    func = tool_call.get("function", {})
                    lines.append(f"{role} [called tool '{func.get('name')}']: {func.get('arguments')}")
            
            # Handle tool results
            elif role == "TOOL":
                result_preview = str(content)[:200] if content else ""
                lines.append(f"TOOL RESULT: {result_preview}")

        return "\n".join(lines)
