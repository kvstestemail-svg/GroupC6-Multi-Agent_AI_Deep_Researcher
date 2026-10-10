# Multi-Agent Deep Researcher Architecture

## System Design Principle
The application is built on LangGraph utilizing a **Hierarchical Sub-Graph pattern**. This resolves common schema validation bugs (such as the standard `400 Invalid Request Error` caused by orphaned tool calls) by fully isolating tool execution from the global routing state.

## The Agent Team

### 🧭 1. Supervisor Agent (The Router)
- **Role:** Central orchestrator.
- **Function:** Dictates the workflow progression. It strictly forces sequential execution: `Retriever -> Critical Analysis -> Insight Generation -> Report Builder -> FINISH`. 
- **Token Tracking:** Utilizes structured outputs (`include_raw=True`) to maintain visibility over orchestration token costs.

### 🕵️ 2. Contextual Retriever (Isolated Sub-Graph)
- **Role:** Evidence gatherer.
- **Tools:** 
  - `local_report_search`: FAISS/LlamaIndex vector retrieval for uploaded PDFs.
  - `tavily_search`: Web scraping for real-time news.
  - `arxiv_search`: Academic paper integration.
  - `wikipedia_search`: Definitional context.
- **Safety Mechanism:** Runs an internal ReAct loop. It consumes its own JSON tool requests and tool outputs internally, generating a synthesized string of findings. Only this clean text is appended to the global state, ensuring downstream LLMs never encounter broken tool schemas.

### 🔬 3. Critical Analysis Agent
- **Role:** Cross-examiner.
- **Function:** Compares the Retriever’s gathered evidence to flag contradictions, source biases, and methodological gaps.

### 💡 4. Insight Generation Agent
- **Role:** Forward-looking synthesizer.
- **Function:** Ingests the data and critical critique to predict macro-trends, second-order effects, and strategic hypotheses.

### 📝 5. Report Builder Agent
- **Role:** Final assembler.
- **Function:** Consolidates the pipeline into a comprehensive Markdown report, preserving citations and context.

## State Trackers (`AgentState`)
- `messages`: Core conversation and AI content strings.
- `tool_logs`: Aggregates the internal search logs to display real-time tool queries and source snippets in the Streamlit UI.
- `token_usage`: Accumulates prompt, completion, and total token expenditure securely across all agent invocations.
- `task_summaries`: Generates an end-of-run execution audit in bullet-point format.