# agent/agent.py — Core Agent Loop
#
# CONCEPT: LLM Basics
# The agent uses Claude via the Anthropic API. Each call requires:
#   - model: which LLM to use
#   - system: the agent's persona and instructions
#   - messages: the full conversation history (short-term memory)
#   - tools: available tools the LLM can request
#   - max_tokens: response length limit
#
# CONCEPT: Tool Calling (ReAct Pattern)
# ReAct = Reason + Act
# The agent loop:
#   1. LLM reasons about what to do
#   2. If it needs a tool → outputs a tool_use block
#   3. We execute the tool → get result
#   4. Result fed back → LLM reasons again (step 1)
#   5. Repeat until LLM outputs a plain text response (done)

import json
import openai
from config import MODEL, MAX_TOKENS, SHOW_THINKING, AGENT_NAME
from memory.memory_manager import MemoryManager
from tools.tools import TOOL_DEFINITIONS, execute_tool, increment_message_count
from utils.logger import (
    log_tool_call, log_tool_result, log_thinking,
    log_error, log_agent
)

# ── System Prompt: Defines the agent's persona and behavior ─────
SYSTEM_PROMPT = f"""You are {AGENT_NAME}, a smart and practical personal research assistant.

Your capabilities:
- Answer questions using your knowledge
- Use the calculator tool for any math or numeric reasoning
- Use web_search to find current information, news, or research
- Use save_note to store important findings the user wants to remember
- Use get_session_info to report session statistics

Guidelines:
- Be concise and direct. Don't over-explain.
- Always use the calculator tool for math — never calculate in your head.
- When unsure about current facts, use web_search.
- After saving a note, confirm what was saved.
- If asked about memory or conversation history, summarize what you remember.
- Think step by step before responding to complex questions.

You have short-term memory (conversation history) that allows you to remember
what was discussed earlier in this session. If the conversation was long, you
may have a summarized version of older turns."""


# Convert Anthropic tool definitions to OpenAI format dynamically
OPENAI_TOOLS = []
for tool in TOOL_DEFINITIONS:
    OPENAI_TOOLS.append({
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"]
        }
    })


class ResearchAgent:
    """
    The main agent that coordinates:
    - LLM calls (via OpenAI API)
    - Tool execution
    - Memory management
    """

    def __init__(self, api_key: str):
        # CONCEPT: LLM Basics — Initialize the OpenAI client pointing to Gemini
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.memory = MemoryManager()
        self.max_tool_iterations = 5  # Safety limit to prevent infinite loops

    def chat(self, user_message: str) -> str:
        """
        Main entry point: process one user message and return a response.
        
        Flow:
        user_message → memory → LLM → [tool loop] → final response
        """
        # 1. Add user message to short-term memory
        self.memory.add_user_message(user_message)
        increment_message_count()

        # 2. Run the ReAct agent loop
        final_response = self._run_agent_loop()

        # 3. Check memory and compress if needed
        self.memory.check_and_compress(self.client)

        return final_response

    def _run_agent_loop(self) -> str:
        """
        ReAct loop: Reason → Act → Observe → Reason → ...
        
        The LLM either:
        (a) Returns text → we're done, return it
        (b) Requests tool use → execute tool, feed result back, repeat
        """
        iteration = 0

        while iteration < self.max_tool_iterations:
            iteration += 1

            if SHOW_THINKING and iteration > 1:
                log_thinking(f"Agent reasoning (iteration {iteration})...")

            # ── Call the LLM ──────────────────────────────────────
            # CONCEPT: LLM Basics — This is the core API call.
            # We pass: model, system prompt, full history, and tool schemas.
            try:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.memory.get_messages()
                response = self.client.chat.completions.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    tools=OPENAI_TOOLS,          # Tell LLM what tools exist
                    messages=messages            # Full conversation history
                )
            except openai.AuthenticationError:
                return "❌ Invalid API key. Please check your OPENAI_API_KEY."
            except openai.RateLimitError as e:
                return f"❌ Rate limit hit. OpenAI says: {str(e)}"
            except Exception as ex:
                log_error(str(ex))
                return f"❌ API error: {str(ex)}"

            # ── Inspect the response ─────────────────────────────
            message = response.choices[0].message

            # CASE 1: LLM wants to use a tool
            if message.tool_calls:
                # Save the assistant's tool-use request to memory
                self.memory.add_assistant_message(message)

                # Process all tool calls in this response
                # (LLM can request multiple tools in one turn)
                tool_results_added = False
                for tool_call in message.tool_calls:
                    tool_result = self._execute_tool_call(tool_call)

                    # Add tool result to memory for next LLM call
                    self.memory.add_tool_result(tool_call.id, tool_result)
                    tool_results_added = True

                if not tool_results_added:
                    return "Unexpected response format from LLM."

                # Loop back — let LLM reason about the tool result

            # CASE 2: LLM is done — extract and return text response
            else:
                text_response = message.content or ""
                # Save assistant response to memory
                self.memory.add_assistant_message(message)
                return text_response

        # Safety: if we hit the iteration limit
        return "⚠️ Agent reached maximum iterations. The task may be too complex — please try rephrasing."

    def _execute_tool_call(self, tool_call) -> str:
        """Execute a single tool call from the LLM."""
        tool_name = tool_call.function.name
        try:
            tool_input = json.loads(tool_call.function.arguments)
        except Exception as ex:
            return f"Error parsing tool arguments: {str(ex)}"

        # Log the tool call for visibility
        log_tool_call(tool_name, tool_input)

        # Execute the tool
        result = execute_tool(tool_name, tool_input)

        # Log the result
        log_tool_result(tool_name, result)

        return result

    def get_memory_stats(self) -> dict:
        """Return current memory statistics."""
        return self.memory.get_stats()

    def reset(self):
        """Clear memory and start a new session."""
        self.memory.clear()
