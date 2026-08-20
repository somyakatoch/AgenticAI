"""
Tool definitions for the agentic AI assistant.

Add your own tools by writing a function, decorating it with @tool,
giving it a clear docstring (the LLM reads this to decide when to use it),
and adding it to the TOOLS list at the bottom.
"""

import ast
import operator
import os
import subprocess

from langchain_core.tools import tool

# Sandbox directory the agent is allowed to read/write in.
WORKSPACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)


def _safe_path(filename: str) -> str:
    """Resolve a filename to a path inside WORKSPACE_DIR, blocking path traversal."""
    path = os.path.abspath(os.path.join(WORKSPACE_DIR, filename))
    if not path.startswith(os.path.abspath(WORKSPACE_DIR)):
        raise ValueError("Access outside the workspace directory is not allowed.")
    return path


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Search the web for current information (news, facts, prices, etc.)
    and return a short summary of the top results. Use this when you need
    up-to-date or real-world information you don't already know."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "web_search is unavailable: run `pip install duckduckgo-search` first."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"No results found for '{query}'."
        lines = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"- {title}: {body} ({href})")
        return "\n".join(lines)
    except Exception as e:
        return f"web_search failed: {e}"


# ---------------------------------------------------------------------------
# Calculator (safe expression evaluation, no eval())
# ---------------------------------------------------------------------------
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '47 * 812 + 3'. Supports
    + - * / // % ** and parentheses. Use this instead of doing arithmetic
    yourself to avoid mistakes."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"calculator error: {e}"


# ---------------------------------------------------------------------------
# File I/O (sandboxed to workspace/)
# ---------------------------------------------------------------------------
@tool
def read_file(filename: str) -> str:
    """Read and return the text contents of a file in the workspace directory."""
    try:
        path = _safe_path(filename)
        if not os.path.exists(path):
            return f"File '{filename}' does not exist in the workspace."
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"read_file error: {e}"


@tool
def write_file(filename: str, content: str) -> str:
    """Write text content to a file in the workspace directory, creating or
    overwriting it. Use this to save results, notes, or generated content."""
    try:
        path = _safe_path(filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to '{filename}'."
    except Exception as e:
        return f"write_file error: {e}"


# ---------------------------------------------------------------------------
# Python execution (subprocess, timeout, no shared state)
# ---------------------------------------------------------------------------
@tool
def run_python(code: str) -> str:
    """Execute a short Python snippet and return its stdout/stderr. Useful
    for quick computations, data manipulation, or checking logic. There is
    no persistent state between calls and no internet access inside the
    snippet."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=WORKSPACE_DIR,
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\n[stderr]\n{result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "run_python error: execution timed out after 10 seconds."
    except Exception as e:
        return f"run_python error: {e}"


# --- add your own tools below ---


TOOLS = [web_search, calculator, read_file, write_file, run_python]
