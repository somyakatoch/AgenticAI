"""
Agentic AI Assistant — a ReAct-style agent built with LangGraph.

The agent reasons about the user's request, decides whether it needs a tool,
calls the tool, observes the result, and loops until it can give a final
answer. Conversation memory persists for the life of the process (per
thread_id), so follow-up questions have context.

This version runs on a LOCAL model via Ollama — no API key, no cost.
Run: python agent.py

--- Switching back to Claude later ---
If you ever get an Anthropic API key, switch providers by:
  1. pip install langchain-anthropic
  2. Set MODEL_PROVIDER = "anthropic" below
  3. Add ANTHROPIC_API_KEY to your .env file
"""

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOLS

load_dotenv()

# "ollama" (free, local) or "anthropic" (needs ANTHROPIC_API_KEY in .env)
MODEL_PROVIDER = "ollama"
OLLAMA_MODEL = "llama3.1"

SYSTEM_PROMPT = """You are a capable, careful agentic assistant.

You have access to tools: web_search, calculator, read_file, write_file,
run_python. Use them whenever they'd make your answer more accurate or
useful — don't guess at facts you could look up, and don't do arithmetic
in your head when you can verify it with the calculator.

Think step by step about what the user actually needs before acting.
If a task needs multiple tool calls in sequence, do them one at a time and
use each result to inform the next step. When you have enough information,
give a clear, direct final answer — don't call tools you don't need."""


def build_llm():
    if MODEL_PROVIDER == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            print("ERROR: langchain-ollama not installed. Run:\n"
                  "  pip install langchain-ollama")
            sys.exit(1)
        return ChatOllama(model=OLLAMA_MODEL, temperature=0)

    elif MODEL_PROVIDER == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set. Add it to your .env file. "
                  "Get one at https://console.anthropic.com/settings/keys")
            sys.exit(1)
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            print("ERROR: langchain-anthropic not installed. Run:\n"
                  "  pip install langchain-anthropic")
            sys.exit(1)
        return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

    else:
        print(f"ERROR: unknown MODEL_PROVIDER '{MODEL_PROVIDER}'. Use 'ollama' or 'anthropic'.")
        sys.exit(1)


def build_agent():
    llm = build_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: MessagesState):
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def main():
    if MODEL_PROVIDER == "ollama":
        print("Checking Ollama connection...")
        try:
            import ollama as ollama_client
            models = [m["model"] for m in ollama_client.list().get("models", [])]
            if not any(OLLAMA_MODEL in m for m in models):
                print(f"WARNING: model '{OLLAMA_MODEL}' not found locally. "
                      f"Run `ollama pull {OLLAMA_MODEL}` first.\nFound: {models}")
        except Exception as e:
            print(f"WARNING: couldn't reach Ollama ({e}). "
                  f"Make sure the Ollama app is running, then try again.")

    app = build_agent()
    thread_config = {"configurable": {"thread_id": "cli-session"}}

    print(f"Agentic AI Assistant (running on {MODEL_PROVIDER}:"
          f"{OLLAMA_MODEL if MODEL_PROVIDER == 'ollama' else 'claude-sonnet-4-6'})")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not user_input:
            continue

        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=thread_config,
            )
            final_message = result["messages"][-1]
            print(f"\nAgent: {final_message.content}\n")
        except Exception as e:
            print(f"\n[Error running agent: {e}]\n"
                  f"If this mentions tool calling, your model may not support "
                  f"it well — try `ollama pull qwen2.5:7b` and set "
                  f"OLLAMA_MODEL = 'qwen2.5:7b' at the top of agent.py.\n")


if __name__ == "__main__":
    main()
