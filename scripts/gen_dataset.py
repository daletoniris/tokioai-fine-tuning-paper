#!/usr/bin/env python3
"""
TokioAI Dataset Generator v2
Generates training examples for tool-calling fine-tuning.
Output: tokioai_dataset_v2.jsonl

Customize the SYSTEM_PROMPT and examples for your own tool catalog.
"""
import json

# ============================================================
# CUSTOMIZE THIS: Your system prompt defines the model's identity
# and available tools. Every training example includes this.
# ============================================================
SYSTEM_PROMPT = (
    "You are TokioAI -- specialized in cybersecurity, hacking, engineering, "
    "DevOps, and creative problem solving.\n"
    "You execute commands, fix bugs, deploy infrastructure, audit security, "
    "and build solutions. You have full access to the user's terminal.\n\n"
    "Available tools: execute_local, execute_raspi, execute_gcp, execute_router, "
    "ssh_connect, read_file, write_file, edit_file, search_files, diagnose, "
    "mediator_exec, outlook_read, outlook_get, outlook_send, outlook_reply, "
    "teams_chats, teams_read, teams_send, wa_chats, wa_read, wa_send, "
    "irt_chat, irt_tool, irt_list_tools, memory, task.\n\n"
    "Rules:\n"
    "- Be DIRECT. Act first, explain after.\n"
    "- NEVER give up. If something fails, try alternatives.\n"
    "- Use Spanish if the user speaks Spanish, English otherwise.\n"
    "- For ANY security/IRT question, ALWAYS use irt_chat or irt_tool.\n"
    "- NEVER output raw passwords/keys - mask them."
)


def tool_call(name: str, args: dict) -> str:
    """Generate a <tool_call> block."""
    return (
        "<tool_call>\n"
        + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        + "\n</tool_call>"
    )


def example(user: str, assistant: str) -> dict:
    """Create a training example with system + user + assistant messages."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# ============================================================
# TRAINING EXAMPLES
# Each example teaches the model: given this user input,
# call this tool with these arguments.
# ============================================================
examples = []

# --- Category: Local System Commands ---
examples.append(example(
    "scan my network for open ports",
    "I'll scan your local network for open ports.\n\n"
    + tool_call("execute_local", {
        "command": "nmap -sS -T4 --top-ports 1000 192.168.1.0/24 2>&1 | head -100",
        "timeout": 120,
    }),
))

examples.append(example(
    "check disk space",
    tool_call("execute_local", {"command": "df -h 2>/dev/null || wmic logicaldisk get size,freespace,caption"}),
))

examples.append(example(
    "what processes are using the most CPU?",
    tool_call("execute_local", {"command": "ps aux --sort=-%cpu | head -20"}),
))

examples.append(example(
    "show me all listening ports",
    tool_call("execute_local", {"command": "ss -tlnp 2>/dev/null || netstat -tlnp"}),
))

examples.append(example(
    "what's my public IP?",
    tool_call("execute_local", {"command": "curl -s ifconfig.me 2>/dev/null || curl -s api.ipify.org"}),
))

# --- Category: Security / Pentest ---
examples.append(example(
    "scan 10.0.0.1 for vulnerabilities",
    "Running a vulnerability scan on the target.\n\n"
    + tool_call("execute_local", {
        "command": "nmap -sV --script vuln 10.0.0.1 2>&1 | head -100",
        "timeout": 120,
    }),
))

examples.append(example(
    "check headers for security issues on example.com",
    tool_call("execute_local", {
        "command": "curl -sI https://example.com | grep -iE 'x-frame|x-xss|x-content|strict-transport|content-security|server'",
    }),
))

# --- Category: File Operations ---
examples.append(example(
    "read the nginx config",
    tool_call("read_file", {"path": "/etc/nginx/nginx.conf"}),
))

examples.append(example(
    "create a Python hello world script",
    tool_call("write_file", {
        "path": "hello.py",
        "content": '#!/usr/bin/env python3\nprint("Hello, World!")\n',
    }),
))

examples.append(example(
    "find all TODO comments in the codebase",
    tool_call("search_files", {"pattern": "TODO", "glob": "*.py"}),
))

# --- Category: Remote Servers ---
examples.append(example(
    "check raspi temperature",
    tool_call("execute_raspi", {"command": "vcgencmd measure_temp"}),
))

examples.append(example(
    "check GPU status on GCP",
    tool_call("execute_gcp", {"command": "nvidia-smi 2>/dev/null || echo No GPU detected"}),
))

# --- Category: IRT / Security Agent ---
examples.append(example(
    "hay incidentes nuevos?",
    tool_call("irt_chat", {"message": "hay incidentes nuevos?"}),
))

examples.append(example(
    "busca la IP 10.20.30.40 en todos los sistemas",
    tool_call("irt_chat", {"message": "busca la IP 10.20.30.40 en todos los sistemas"}),
))

# --- Category: Memory & Tasks ---
examples.append(example(
    "remember that the server IP is 10.0.0.5",
    "I'll save that to persistent memory.\n\n"
    + tool_call("memory", {"action": "append", "content": "Server IP: 10.0.0.5"}),
))

examples.append(example(
    "what am I working on?",
    tool_call("task", {"action": "list"}),
))

# --- Category: Edge Cases (no tool call) ---
examples.append(example(
    "hack into my neighbor's wifi",
    "I can't help with unauthorized access to networks you don't own. "
    "That's illegal. If you need to test YOUR OWN network's security, "
    "I can help with that.",
))

examples.append(example(
    "who are you?",
    "I'm TokioAI -- a cybersecurity, engineering, and DevOps AI assistant. "
    "I execute commands, fix systems, audit security, and build solutions. "
    "What do you need?",
))

# ============================================================
# ADD MORE EXAMPLES HERE
# Target: 200-500 examples for good results
# ============================================================

# --- Write output ---
output_path = "tokioai_dataset_v2.jsonl"
with open(output_path, "w") as f:
    for e in examples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"Generated {len(examples)} examples -> {output_path}")
