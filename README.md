# DHL Strategic Intelligence Agent

An AI-powered Strategic Intelligence Agent that collects live information
about DHL Group, stores and analyzes it, reasons over it using an explicit
agent workflow, and generates evidence-backed executive recommendations
through an interactive, conversational dashboard.

> **Core question this system answers:** *"If you were the CEO of DHL today,
> what would you do next and why?"*

---

## 1. System Architecture

```
+-------------------------------------------------------------------+
|                        DATA COLLECTION LAYER                      |
|   ddgsdhl.ipynb        Newsapi.ipynb        Supplychain.ipynb     |
|   (DuckDuckGo Search)  (NewsAPI)            (web scrape)          |
|        |                    |                     |               |
|   ddgs_dhl.csv         dhl_news.json    supplychaindive_articles  |
|        +--------------------+---------------------+               |
+---------------------------+----------------------------------------+
                             v
+-------------------------------------------------------------------+
|                     PREPROCESSING LAYER                           |
|                     Preprocessing.ipynb                           |
|   merge -> drop_duplicates(title) -> fillna -> clean_text()       |
|   -> lowercase, strip URLs/punctuation -> remove stopwords (NLTK) |
|                  processed_dhl_dataset.csv                        |
+---------------------------+----------------------------------------+
                             v
+-------------------------------------------------------------------+
|                  KNOWLEDGE REPOSITORY LAYER                       |
|                     Embeddings.ipynb                              |
|   SentenceTransformer("all-MiniLM-L6-v2")                         |
|   -> ChromaDB PersistentClient -> collection "dhl_intelligence"   |
+---------------------------+----------------------------------------+
                             v
+-------------------------------------------------------------------+
|                      AGENT ORCHESTRATOR                           |
|                       dashboard.py                                |
|                                                                     |
|   GOAL -> PLAN -> RETRIEVE -> [SEARCH if needed] -> ANALYZE       |
|        -> DECIDE -> RECOMMEND -> VALIDATE                         |
|                                                                     |
|   - set_goal()              explicit objective                   |
|   - plan_investigation()    LLM decides which categories/queries |
|   - retrieve_evidence()     calls the Strategic Intelligence      |
|                              Engine above                          |
|   - decide_search_need()    autonomous tool-selection: is the    |
|                              local knowledge base enough, or is   |
|                              live web search warranted?            |
|   - web_search_tool()       DuckDuckGo (ddgs) live search         |
|   - analyze_evidence()      groups/summarizes retrieved entries  |
|   - decide_priorities()     explicit ranking by severity x conf. |
|   - recommend()             LLM-generated executive briefing      |
|   - validate_recommendation() structural + grounding checks,     |
|                              regenerates once if checks fail      |
+---------------------------+----------------------------------------+
                             v
+-------------------------------------------------------------------+
|                       LOCAL LLM (transformers)                    |
|            Mistral 7B Instruct -- runs entirely on-device         |
|   - Sentiment classification (structured JSON output)             |
|   - Investigation planning (autonomous category/query selection)  |
|   - Executive briefing generation                                 |
+---------------------------+----------------------------------------+
                             v
+-------------------------------------------------------------------+
|              EXECUTIVE INTELLIGENCE DASHBOARD                     |
|                  Streamlit -- dashboard.py                        |
|   1. Company Overview       6. Sentiment Analysis                 |
|   2. Market Intelligence    7. Strategic Recommendations          |
|   3. Opportunity Monitor    8. CEO Briefing                       |
|   4. Risk Monitor           9. Ask the Agent (conversational)     |
|   5. Trend Monitor                                                 |
+-------------------------------------------------------------------+
```

## 2. Data Flow Diagram

```
[Web Sources] -> [Raw collection (3 CSV/JSON files)] -> [Standardize schema]
   -> [Merge] -> [Deduplicate] -> [Clean text] -> [Remove stopwords]
   -> processed_dhl_dataset.csv
   -> [Embed: all-MiniLM-L6-v2] -> [Index: ChromaDB]
   -> [Retrieve: semantic search per topic query]
   -> [Score: occurrence-weighted, 7 categories]
        +-> [Frame: Opportunity / Trend]
        +-> [Corpus-wide direct keyword scan -> Risk Monitor]
   -> [Agent: Goal -> Plan -> Retrieve -> (Search) -> Analyze -> Decide
              -> Recommend -> Validate]
   -> [Present: Streamlit dashboard, 9 sections, including live chat]
```

## 3. Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Data collection | `ddgs`, NewsAPI, `requests` + `BeautifulSoup` | 3 independent public sources, satisfies minimum source requirement |
| Data processing | `pandas`, `re`, `nltk` (stopwords) | Standard, lightweight cleaning pipeline |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` | Brief-approved; small, fast, no GPU required |
| Knowledge repository | `chromadb` (PersistentClient) | Brief-approved; simple local persistence, built-in similarity search |
| Reasoning / LLM | `transformers` — **Mistral 7B Instruct** (local) | Brief-approved, open-source, runs entirely offline. Chosen over Llama 3.1 specifically because Mistral is **not gated** on Hugging Face — no access request/approval delay |
| Agent orchestration | Plain Python (no agent framework) | Explicit, inspectable Goal→Plan→Retrieve→Analyze→Decide→Recommend→Validate workflow, each stage independently testable and printable |
| Live search tool | `ddgs` (DuckDuckGo) | Second, genuinely distinct tool available to the agent beyond the local knowledge base |
| Dashboard | `streamlit` + `plotly` | Brief-approved; native chat components (`st.chat_input`) used for the conversational interface |

**Explicitly not used as the reasoning engine:** OpenAI, Anthropic, or Gemini
APIs, per the brief's mandatory constraint. **Also not used:** Ollama — the
system was originally built on Ollama, then migrated to direct
`transformers` model loading for finer control over generation parameters;
both are equally brief-compliant, the migration was a tooling preference,
not a compliance requirement.

## 4. Design Decisions

| Decision | Alternative considered | Why this choice won |
|---|---|---|
| **Occurrence-weighted keyword scoring** instead of presence-only | Boolean "does this keyword appear" check | Presence-only let one incidental mention dominate over a theme mentioned 7+ times in the same evidence set |
| **Multi-category scoring instead of if/elif chains** | First-match-wins branching | if/elif order determined the outcome, not the actual evidence strength |
| **Regex word-boundary matching for short/ambiguous keywords** | Plain substring matching everywhere | The bare keyword "robot" matched inside "HappyRobot" (an AI agent product name), wrongly inflating automation scores for AI-agent articles |
| **Separate corpus-wide keyword scan for Risk Monitor** | Reuse the same per-query semantic retrieval used for Opportunities | Diagnosed that embedding similarity favored brand terms ("DHL") over topic terms ("tariff"), causing real risk content (confirmed: 17 documents genuinely mention "tariff") to be missed by semantic top-8 retrieval |
| **Explicit agent orchestration layer (Goal→...→Validate)** | A single prompt→LLM→response pipeline | Demonstrates planning, autonomous decision-making, tool use beyond the LLM itself, and validation before presenting — not just RAG |
| **Autonomous search-or-not decision** (`decide_search_need`) | Always searching, or never searching | The agent judges for itself whether local retrieval coverage was sufficient before deciding to invoke a second tool, rather than following a fixed rule |
| **Validation step that can trigger regeneration** | Generate and present the briefing directly | Checks structural completeness (3 required sections) and evidence grounding before showing the user anything, regenerating once if checks fail |
| **Mistral 7B over Llama 3.1** | Llama 3.1 8B (also brief-approved) | Mistral is not a gated model on Hugging Face — no access request/approval wait |
| **`float16` + reduced token budget on CPU** | Default `float32`, unconstrained generation length | CPU inference is memory-bandwidth-bound; halving precision and capping output length materially reduces response latency with limited quality cost for short, structured outputs |

## 5. AI Pipeline — How a Question Gets Answered

This describes the live, conversational path (Dashboard Section 9 — "Ask
the Agent"), which is the fullest expression of the pipeline:

1. **Goal**: the user's typed question becomes the agent's explicit objective.
2. **Plan**: the LLM is asked to decide which 3-4 categories are most relevant to that goal and to generate a specific retrieval query for each — autonomous decision-making, not a hardcoded query list.
3. **Retrieve**: each planned query is run against the ChromaDB collection via semantic similarity search (top-8 results), and the corpus-wide risk scan runs independently.
4. **Decide search need**: the agent compares how much evidence local retrieval actually returned against what was planned. If coverage is low, it autonomously invokes a second tool — a live DuckDuckGo search — to fill the gap.
5. **Analyze**: retrieved entries are grouped into Opportunity / Risk / Trend and summarized.
6. **Decide priorities**: opportunities and risks are explicitly ranked by a combined severity × confidence score; only the top-ranked items are carried forward.
7. **Recommend**: the LLM synthesizes the decided priorities into a structured "what happened / why it matters / what to do next" briefing.
8. **Validate**: the briefing is checked for structural completeness and evidence grounding before being shown to the user. If either check fails, the agent regenerates once with corrective context rather than silently presenting an invalid result.

## 6. Known Limitations

- **Competitor Activity** (Section 2): the corpus is collected primarily via DHL-branded search queries, so it is heavily DHL-press-skewed. Direct queries for competitor content (FedEx/UPS) return sparse results.
- **Public Sentiment**: no community/social data source (Reddit, forums) is currently in the collection pipeline — only press/news sources.
- **Sentiment Trend over time**: requires normalized `published_at` timestamps across all three sources; only NewsAPI provides these consistently.
- **CPU inference latency**: without a GPU, local LLM generation (planning, briefing, sentiment) is noticeably slower than a hosted API would be. Mitigated via `float16` precision and a reduced output token budget, but not eliminated — a single agent run (plan + retrieve + recommend) can take tens of seconds on CPU-only hardware.
- **Embeddings pass-through**: the explicitly computed `all-MiniLM-L6-v2` embeddings are not passed directly into ChromaDB's storage call; ChromaDB falls back to its own internal default embedding function (also MiniLM-family, so limited practical impact, but a known gap between intent and implementation).

## 7. How to Run

```bash
# 1. Collect data (run once, or periodically to refresh)
jupyter notebook ddgsdhl.ipynb        # -> ddgs_dhl.csv
jupyter notebook Newsapi.ipynb        # -> dhl_news.json
jupyter notebook Supplychain.ipynb    # -> supplychaindive_articles.csv

# 2. Process and index
jupyter notebook Preprocessing.ipynb  # -> processed_dhl_dataset.csv
jupyter notebook Embeddings.ipynb     # -> ChromaDB collection "dhl_intelligence"

# 3. Install dependencies
pip install streamlit chromadb sentence-transformers pandas plotly ddgs
pip install transformers accelerate torch

# 4. Launch the dashboard (downloads Mistral 7B on first run, ~15GB)
streamlit run dashboard.py
```

## 8. Project File Map

| File | Role |
|---|---|
| `ddgsdhl.ipynb` | Data source 1: DuckDuckGo search collection |
| `Newsapi.ipynb` | Data source 2: NewsAPI collection |
| `Supplychain.ipynb` | Data source 3: SupplyChainDive scrape |
| `Preprocessing.ipynb` | Merge, dedupe, clean, stopword removal |
| `Embeddings.ipynb` | Embedding generation + ChromaDB indexing |
| `dashboard.py` | **Main deliverable** — strategic intelligence engine, agent orchestrator, search tool, and 9-section Streamlit dashboard (including conversational chat interface) |
| `Agent_orchestrator.ipynb` | Standalone reference copy of the agent workflow logic, for readability outside the full dashboard file |
