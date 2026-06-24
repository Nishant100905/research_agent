#!/usr/bin/env python3
# main.py — Entry point for the Personal Research Assistant Agent
#
# Run with: python main.py
# Requires: GEMINI_API_KEY environment variable

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.agent import ResearchAgent
from config import MODEL
from utils.logger import (
    log_user, log_agent, log_header, log_divider,
    log_memory_event, log_error, Colors
)


HELP_TEXT = f"""
{Colors.CYAN}Available commands:{Colors.RESET}
  {Colors.BOLD}/help{Colors.RESET}    - Show this help
  {Colors.BOLD}/memory{Colors.RESET}  - Show memory stats
  {Colors.BOLD}/reset{Colors.RESET}   - Clear memory and start fresh
  {Colors.BOLD}/quit{Colors.RESET}    - Exit the agent

{Colors.CYAN}Example prompts:{Colors.RESET}
  "What is 23% of 4500?"
  "Search for latest Python 3.13 features"
  "Save a note: Review transformer architecture paper"
  "What topics have we discussed so far?"
  "How long have we been chatting?"
"""

DEMO_PROMPTS = [
    "What is 15% of 847 plus 230?",
    "Search for information about AI agents",
    "Save a note titled 'Agent Architecture' with content: ReAct pattern uses Reason+Act loops",
    "What session statistics do we have?",
    "Explain what short-term memory means for LLMs in simple terms",
]


def print_demo_prompts():
    """Show example prompts to get started."""
    print(f"\n{Colors.DIM}💡 Try these example prompts:{Colors.RESET}")
    for i, prompt in enumerate(DEMO_PROMPTS, 1):
        print(f"  {Colors.DIM}{i}. {prompt}{Colors.RESET}")
    print()


def get_api_key() -> str:
    """Get API key from environment or prompt user."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        print(f"{Colors.GREEN}✓ API key loaded from environment.{Colors.RESET}")
        return api_key

    print(f"\n{Colors.YELLOW}⚠️  GEMINI_API_KEY not found in environment.{Colors.RESET}")
    print("Get your free key at: https://aistudio.google.com/app/apikey")
    api_key = input("Enter your API key (or press Enter to use demo mode): ").strip()

    if not api_key:
        print(f"\n{Colors.YELLOW}Running in DEMO MODE — API calls will fail without a real key.{Colors.RESET}")
        print("Set it with: export GEMINI_API_KEY=your_key_here\n")
        return "demo_key"

    return api_key


def handle_command(command: str, agent: ResearchAgent) -> bool:
    """Handle slash commands. Returns True if command was handled."""
    cmd = command.lower().strip()

    if cmd == "/help":
        print(HELP_TEXT)
        return True

    elif cmd == "/memory":
        stats = agent.get_memory_stats()
        print(f"\n{Colors.MAGENTA}🧠 Memory Statistics:{Colors.RESET}")
        print(f"  Messages in memory : {stats['current_messages']}")
        print(f"  Total turns ever   : {stats['total_turns']}")
        print(f"  Has summary        : {stats['has_summary']}")
        print(f"  Times summarized   : {stats['summarizations']}")
        if stats['summary_preview']:
            print(f"  Summary preview    : {stats['summary_preview']}")
        return True

    elif cmd == "/reset":
        agent.reset()
        print(f"{Colors.GREEN}✓ Memory cleared. Starting fresh session.{Colors.RESET}")
        return True

    elif cmd in ("/quit", "/exit", "/q"):
        print(f"\n{Colors.CYAN}👋 Goodbye! Your notes have been saved to agent_notes.txt{Colors.RESET}\n")
        sys.exit(0)

    return False


def run_demo_mode():
    """Run through demo prompts automatically (for testing without API key)."""
    print(f"\n{Colors.YELLOW}=== DEMO MODE — showing what the agent would do ==={Colors.RESET}\n")

    demos = [
        ("Calculator", "What is sqrt(144) + 15% of 500?", 
         "🔧 Tool Call: calculator\n   ↳ expression: sqrt(144) + 500 * 0.15\n📦 Result: 12 + 75 = 87.0\n\nThe answer is 87.0"),
        
        ("Web Search", "Search for Python agent frameworks",
         "🔧 Tool Call: web_search\n   ↳ query: Python agent frameworks\n📦 Found 3 results...\n\nThe top Python agent frameworks include LangChain, LlamaIndex, and DSPy..."),
        
        ("Save Note", "Save a note: LLM agents use tool calling + memory",
         "🔧 Tool Call: save_note\n   ↳ title: LLM Agent Pattern\n   ↳ content: LLM agents use tool calling + memory\n📦 Note saved!\n\n✅ Saved your note about LLM agents."),
        
        ("Session Info", "How long have we been chatting?",
         "🔧 Tool Call: get_session_info\n📦 Session: 3 messages, 2m 15s\n\nWe've been chatting for about 2 minutes and exchanged 3 messages."),
    ]

    for concept, prompt, response in demos:
        log_divider()
        print(f"{Colors.BOLD}[Demonstrating: {concept}]{Colors.RESET}")
        log_user(prompt)
        print(f"\n  {Colors.YELLOW}🤖 Agent processes...{Colors.RESET}")
        print(f"\n  {response}")
        print()


def main():
    """Main CLI loop."""
    log_header()

    # Get API key
    api_key = get_api_key()

    # If no real key, show demo and exit
    if api_key == "demo_key":
        run_demo_mode()
        print(f"\n{Colors.CYAN}To run the real agent, set GEMINI_API_KEY and run again.{Colors.RESET}\n")
        return

    # Initialize agent
    print(f"\n{Colors.GREEN}✓ Agent initialized with:{Colors.RESET}")
    print(f"  Model  : {MODEL}")
    print(f"  Tools  : calculator, web_search, save_note, get_session_info")
    print(f"  Memory : short-term (auto-summarizes after {10} turns)")

    agent = ResearchAgent(api_key)
    print_demo_prompts()

    print(f"Type {Colors.BOLD}/help{Colors.RESET} for commands. Type your message and press Enter.\n")
    log_divider()

    # ── Main chat loop ──────────────────────────────────────────
    while True:
        try:
            # Get user input
            user_input = input(f"\n{Colors.BLUE}{Colors.BOLD}You: {Colors.RESET}").strip()

            # Skip empty input
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                handle_command(user_input, agent)
                continue

            # Process message through agent
            response = agent.chat(user_input)
            log_agent(response)
            log_divider()

        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}👋 Session ended. Your notes are in agent_notes.txt{Colors.RESET}\n")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
