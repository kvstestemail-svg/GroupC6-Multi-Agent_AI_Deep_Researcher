# Multi-Agent Deep Researcher

An enterprise-grade, autonomous research system built with LangGraph, LlamaIndex, FAISS, and Streamlit, powered by OpenRouter. The application coordinates specialized AI agents orchestrated by a centralized Supervisor to ingest documents, search live web and academic databases, cross-examine evidence, and synthesize publication-ready reports.

## Authors

- Kiran Viradiya
- Deigratia Daniels
- Nithin Kumar Kandagatla
- Roland Johnson

## Architecture Overview

```text
User Query / PDF Upload
          │
          ▼
   Streamlit UI (app.py)
          │
          ▼
  🧭 Supervisor Agent (State Router)
          │
    ┌─────┴───────────────────────────────────────────────────────┐
    │                                                             │
    ▼                                                             ▼
🕵️ Contextual Retriever (ReAct Sub-Graph)              🔬 Critical Analysis Agent
    ├── local_report_search (FAISS / LlamaIndex)                  │  (Flags bias & contradictions)
    ├── arxiv_search (Academic Papers)                            ▼
    ├── tavily_search (Live Web News)                   💡 Insight Generation Agent
    └── wikipedia_search (Definitions)                            │  (Hypotheses & trends)
                                                                  ▼
                                                        📝 Report Builder Agent
                                                             (Final structured Markdown)
```

The system employs a Hierarchical Sub-Graph Pattern: the Contextual Retriever executes its tool-calling loop internally and returns clean synthesized findings to the global workflow state. This prevents tool-schema pollution and eliminates downstream API schema errors (such as OpenAI 400 `invalid_request_error`).

## Directory Structure

```text
deep_researcher/
├── requirements.txt         # Project dependencies
├── app.py                   # Streamlit application entry point
├── agent.md                 # System architecture documentation
├── README.md                # Project setup and user guide
├── agents/
│   ├── __init__.py
│   ├── state.py             # AgentState schema and token reducers
│   ├── supervisor.py        # Supervisor routing agent
│   ├── workers.py           # Worker node factory functions
│   └── graph.py             # LangGraph StateGraph assembly
├── tools/
│   ├── __init__.py
│   └── retrievers.py        # FAISS indexer and retrieval tools
└── ui/
    ├── __init__.py
    └── components.py        # Streamlit sidebar and chat rendering
```

## Core Capabilities

- **Autonomous Supervisor Routing:** A structured routing agent determines execution flow sequentially without hardcoded keyword conditions.
- **Document Ingestion & Retrieval:** Uploaded PDFs are chunked and embedded via `BAAI/bge-small-en-v1.5` and queried using a local FAISS index managed by LlamaIndex.
- **Multi-Source External Research:** Concurrent search integration across arXiv, Tavily Search, and Wikipedia.
- **Live Execution Telemetry:** Real-time expander in the UI displays supervisor routing decisions, tool calls, search queries, and raw source snippets.
- **Token Consumption Tracking:** Tracks prompt tokens, completion tokens, and total usage individually for each agent, with aggregate session metrics.
- **Task Summary Auditing:** Produces a high-level task breakdown summarizing each agent's contributions upon run completion.

## Prerequisites

- Python 3.10 or higher
- An OpenRouter API Key (supports models with tool calling, e.g., `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`)
- A Tavily API Key (optional, required for live internet searches)

## Installation & Setup

Clone or create the project folder:

```bash
mkdir deep_researcher
cd deep_researcher
```

Create and activate a virtual environment:

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration & Execution

Start the Streamlit application:

```bash
streamlit run app.py
```

Configure credentials in the UI sidebar:

- Enter your OpenRouter API Key.
- Enter your Tavily API Key (if web searching is required).
- Specify the OpenRouter Model ID (defaults to `openai/gpt-4o`).

Index PDF reports (optional):

- Upload single or multiple PDF documents under **Upload Reports for Analysis**.
- Click **Index Documents**.
- Wait for the index confirmation before running queries that reference uploaded material.

Submit a research prompt: enter your query into the chat input field to initiate deep research.

## Example Prompts

**Comparative Document Analysis:**

> Analyze the commercial deployment timelines in my uploaded report against recent papers on arXiv. Identify contradictions between corporate targets and fundamental physical degradation limits.

**Broad Market Research:**

> Provide a deep research report on the state of autonomous AI software engineers. Contrast current benchmark achievements against production failure modes reported in recent industry news.

## Agent Pipeline Details

| Agent | Responsibilities | Output to Global State |
| --- | --- | --- |
| Supervisor | Orchestrates flow; routes tasks sequentially across workers. | Routing key (`next`) |
| Contextual Retriever | Queries FAISS vector stores, arXiv, Wikipedia, and Tavily. | Synthesized evidence with source links |
| Critical Analysis | Identifies source bias, factual tensions, and methodology gaps. | Critical evaluation bullet points |
| Insight Generation | Extrapolates trends, second-order effects, and hypotheses. | Structured strategic insights |
| Report Builder | Compiles research into an executive Markdown report. | Final formatted report deliverable |
