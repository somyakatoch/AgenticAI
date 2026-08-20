# Agentic AI Assistant

A general-purpose agentic AI built with **LangGraph** + **Claude**. It follows a
ReAct-style loop (Reason → Act → Observe) and can autonomously decide which
tools to call, chain multiple tool calls together, and keep conversation
memory across turns.

## Architecture

```
User Input
    │
    ▼
┌─────────────┐      tool call?      ┌──────────────┐
│   Agent     │ ───────────────────▶│  Tool Node    │
│  (Claude)   │◀─────────────────── │ (executes)    │
└─────────────┘     tool result      └──────────────┘
    │
    │ no more tool calls
    ▼
Final Answer
```

This is implemented as a small state graph with two nodes (`agent`, `tools`)
and a conditional edge that loops back to the agent until it stops calling
tools.

## Tools included

| Tool | What it does |
|---|---|
| `web_search` | Searches the web (DuckDuckGo, no API key needed) |
| `calculator` | Evaluates math expressions safely |
| `read_file` / `write_file` | Reads/writes files in a sandboxed `workspace/` folder |
| `run_python` | Executes short Python snippets in a subprocess and returns stdout |

You can add your own tools in `tools.py` — see the `# --- add your own tools below ---` marker.

## Setup (running locally with Ollama — free, no API key)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.1            # one-time model download (~4.7GB)
```

Make sure the Ollama app is running (it usually starts automatically after
install; otherwise run `ollama serve`).

## Switching to Claude later

Local models are noticeably weaker at tool-calling than Claude. If you get
an Anthropic API key later:

1. `pip install langchain-anthropic`
2. In `agent.py`, set `MODEL_PROVIDER = "anthropic"`
3. `cp .env.example .env` and add your `ANTHROPIC_API_KEY`

Get a key at https://console.anthropic.com/settings/keys

## Run

```bash
python agent.py
```

Then chat with it:

```
You: What's 47 * 812, and also find me the current price of Bitcoin?
Agent: [calls calculator] [calls web_search] ...
```

Type `exit` or `quit` to stop.

## Extending this project

- **Add a tool**: write a function in `tools.py`, decorate with `@tool`, add it to the `TOOLS` list.
- **Add memory persistence**: swap the in-memory `MemorySaver` in `agent.py` for `SqliteSaver` to persist across restarts.
- **Add multi-agent collaboration**: split responsibilities into separate LangGraph subgraphs (e.g. a "researcher" and a "writer") and route between them — LangGraph supports nested graphs natively.
- **Add guardrails**: validate tool inputs/outputs before they re-enter the loop (especially for `run_python` and `write_file`).

## Project structure

```
agentic-ai-project/
├── agent.py           # LangGraph agent definition + CLI loop
├── tools.py           # Tool definitions
├── requirements.txt
├── .env.example
├── workspace/          # sandboxed read/write area for the agent
└── README.md
```
