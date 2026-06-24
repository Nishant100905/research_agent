import sys

# Ensure stdout and stderr use UTF-8 encoding on all systems (especially Windows console)
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

class Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    DIM     = "\033[2m"


def log_user(message: str):
    """Display user message."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}👤 You:{Colors.RESET} {message}")


def log_agent(message: str):
    """Display agent final response."""
    print(f"\n{Colors.GREEN}{Colors.BOLD}🤖 ResearchBot:{Colors.RESET} {message}")


def log_tool_call(tool_name: str, inputs: dict):
    """Display when agent calls a tool."""
    print(f"\n  {Colors.YELLOW}🔧 Tool Call:{Colors.RESET} {Colors.BOLD}{tool_name}{Colors.RESET}")
    for k, v in inputs.items():
        print(f"  {Colors.DIM}   ↳ {k}: {v}{Colors.RESET}")


def log_tool_result(tool_name: str, result: str):
    """Display tool result."""
    preview = result[:120] + "..." if len(result) > 120 else result
    print(f"  {Colors.CYAN}📦 Result from {tool_name}:{Colors.RESET} {preview}")


def log_memory_event(event: str):
    """Display memory management events."""
    print(f"\n  {Colors.MAGENTA}🧠 Memory:{Colors.RESET} {event}")


def log_thinking(text: str):
    """Display agent reasoning (if enabled)."""
    print(f"  {Colors.DIM}💭 {text}{Colors.RESET}")


def log_error(text: str):
    """Display errors."""
    print(f"\n  {Colors.RED}❌ Error:{Colors.RESET} {text}")


def log_divider():
    print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")


def log_header():
    print(f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════╗
║       🤖  Personal Research Assistant Agent      ║
║   LLM Basics · Tool Calling · Memory Management  ║
╚══════════════════════════════════════════════════╝
{Colors.RESET}""")


def log_memory_status(turn_count: int, limit: int, summarized: bool):
    """Show current memory state."""
    bar_filled = int((turn_count / limit) * 10)
    bar = "█" * bar_filled + "░" * (10 - bar_filled)
    status = f"[{bar}] {turn_count}/{limit} turns"
    if summarized:
        status += " (older turns summarized)"
    print(f"  {Colors.MAGENTA}📊 Memory Status: {status}{Colors.RESET}")
