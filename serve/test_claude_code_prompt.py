"""Claude Code's system prompt rewritten for the local model (frontend.local_claude_code_system, the config's
"claude_code_prompt")."""
import unittest

from serve import frontend as fe

HEADER = "x-anthropic-billing-header: cc_version=2.1.296.55c; cc_entrypoint=sdk-cli;"
BODY = (
    "\nYou are an agent working with the user toward their goals, using your own judgment along the way.\n\n"
    "IMPORTANT: Assist with authorized security testing, defensive security, CTF challenges, and educational "
    "contexts. Refuse requests for destructive techniques.\n\n"
    "# Harness\n - Text you output outside of tool use is displayed to the user as Github-flavored markdown.\n\n"
    "When you use a pronoun for someone, use they/them.\n\n"
    "# Memory\n\nYou have a persistent file-based memory at `C:\\mem\\dir\\`. This directory already exists.\n\n"
    "```markdown\n---\nname: x\n---\n```\n\n`user`: who the user is.\n\n"
    "# Environment\n - The most recent Claude models are the Claude 5 family.\n"
    " - Claude Code is available as a CLI in the terminal.\n - Fast mode for Claude Code uses Claude Opus.\n\n"
    "# Context management\nWhen the conversation grows long, some of it is summarized.")
SDK = [{"type": "text", "text": HEADER}, {"type": "text", "text": "You are a Claude agent, built on Anthropic's "
                                                                    "Claude Agent SDK."}, {"type": "text", "text": BODY}]


def system_of(blocks, mode="local"):
    old = fe.CLAUDE_CODE_PROMPT
    fe.CLAUDE_CODE_PROMPT = mode
    try:
        return fe.anthropic_to_messages({"system": blocks, "messages": [{"role": "user", "content": "hi"}]})[0][0]["content"]
    finally:
        fe.CLAUDE_CODE_PROMPT = old


class LocalClaudeCodePrompt(unittest.TestCase):
    def test_claude_only_parts_go(self):
        s = system_of(SDK)
        self.assertTrue(s.startswith(fe.LOCAL_AGENT_INTRO))
        for gone in ("billing-header", "Claude agent", "security testing", "pronoun", "Claude 5 family",
                     "Fast mode", "# Environment", "You are an agent working"):
            self.assertNotIn(gone, s)
        self.assertIn("# Harness\n - Text you output", s)
        self.assertIn("# Context management\nWhen the conversation grows long", s)

    def test_memory_keeps_its_path(self):
        s = system_of(SDK)
        self.assertIn("# Memory\nPersistent memory lives in `C:\\mem\\dir\\`", s)
        self.assertNotIn("```markdown", s)

    def test_stable(self):
        # a pure function of the text: the conversation cache sees the same prompt on every turn (the real
        # 2.1.296 SDK prompt went from 1,361 to 676 tokens; this fixture is too short to show it)
        self.assertEqual(system_of(SDK), system_of(SDK))

    def test_as_sent_only_pins_the_stamp(self):
        s = system_of(SDK, "as_sent")
        self.assertTrue(s.startswith(fe.pin_billing_stamp(HEADER)))
        self.assertIn("security testing", s)

    def test_interactive_identity_and_environment(self):
        env = ("\n# Environment\nYou have been invoked in the following environment:\n - Primary working directory: D:\\x\n"
               " - The most recent Claude models are the Claude 5 family.\n\n# Plan mode\nDo not edit files yet.")
        s = system_of(HEADER + "You are Claude Code, Anthropic's official CLI for Claude.\n" + env)
        self.assertNotIn("official CLI", s)
        self.assertIn("# Environment\nYou have been invoked in the following environment:\n - Primary working "
                      "directory: D:\\x", s)
        self.assertNotIn("Claude 5 family", s)
        self.assertIn("# Plan mode\nDo not edit files yet.", s)     # a section it does not know passes as sent

    def test_other_prompts_untouched(self):
        self.assertEqual(system_of("You are a helpful assistant."), "You are a helpful assistant.")
        other = HEADER + "Some other agent's prompt."             # a header without Claude Code's identity
        self.assertEqual(system_of(other), fe.pin_billing_stamp(other))


if __name__ == "__main__":
    unittest.main()
