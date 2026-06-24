# config.py — Central configuration for the Research Agent

# ── Model Settings ──────────────────────────────────────────────
MODEL = "gemini-3.5-flash"                    # LLM to use
MAX_TOKENS = 1024                      # Max tokens per response
TEMPERATURE = 0.7                      # Creativity (0=deterministic, 1=creative)

# ── Memory Management ───────────────────────────────────────────
SHORT_TERM_MEMORY_LIMIT = 10          # Max conversation turns before summarization
SUMMARY_KEEP_RECENT = 4              # How many recent turns to keep after summarizing
TOKEN_WARNING_THRESHOLD = 6000       # Warn user when approaching token limit

# ── Tool Settings ───────────────────────────────────────────────
NOTES_FILE = "agent_notes.txt"        # File where save_note tool writes

# ── Display ─────────────────────────────────────────────────────
AGENT_NAME = "ResearchBot"
SHOW_THINKING = True                  # Show tool calls and memory events in terminal
