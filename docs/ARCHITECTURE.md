# Architecture

Drawn from the code as it is, not from the plan. Every box is a module in `backend/`;
arrows are calls. Solid boxes run on this machine; nothing leaves it unless the router
chooses web search.

```mermaid
flowchart TD
  UI["Next.js frontend<br/>QueryPanel: chat, uploads, approvals, pipeline timeline"]
  API["FastAPI routes<br/>/api/stream (SSE) · /api/ingest · /api/documents · /api/actions · /api/traces"]
  UI -->|query, history, conversation_id, images| API
  API --> ORCH

  subgraph ORCH["MasterOrchestrator  (orchestrator/master_llm.py)"]
    direction TB
    TRACE["RequestTrace<br/>t_ms on every event, trace per request"]
    STATE["ConversationStore<br/>per-conversation image context"]
    REWRITE["history rewrite<br/>(only with history; clarification guard)"]
    ROUTER["AgentRouter<br/>1 deterministic rules → 2 knowledge-base probe → 3 LLM classifier"]
    REWRITE --> ROUTER
  end

  ROUTER -->|Search_Knowledge_Base| KB
  ROUTER -->|Web_Search / Visualize_Data without numbers| WEB
  ROUTER -->|Visualize_Data with numbers| VIZD["VisualizerAgent<br/>extract points → fixed renderer"]
  ROUTER -->|Send_Email / Send_Telegram / Workspace_Task| ACT
  ROUTER -->|Read_Email| GMAIL
  ROUTER -->|Vision_Analysis| IMG["ImageAnalyzer<br/>EasyOCR text + LLaVA description"]
  ROUTER -->|Direct_Chat / Ambiguous_Query| GEN

  subgraph KB["Knowledge-base path"]
    direction TB
    HYB["VectorDatabase.hybrid_retrieve<br/>FAISS dense ∪ BM25 → reciprocal rank fusion (top-10)"]
    RR["RerankerModel<br/>cross-encoder → top-5"]
    HYB --> RR
  end

  subgraph WEB["Web path  (retrieval/web_search.py)"]
    direction TB
    Q["build_search_queries<br/>raw first · per-period queries · expansions"]
    T1["SearXNG (self-hosted metasearch)"]
    T2["DuckDuckGo HTML<br/>challenge detected → degraded flag"]
    T3["Wikipedia + Google News"]
    FETCH["fetch pool<br/>HTML extractor · pypdf + table annotation · filing-first · slow-host memory"]
    MERGE["merge: dedupe, cap 16k chars"]
    SER["series_from_context<br/>chart figures from annotated tables"]
    Q --> T1 --> T2 --> T3 --> FETCH --> MERGE --> SER
  end

  KB --> GEN
  MERGE --> GEN
  GEN["GenerationModel  (qwen2.5:3b via Ollama, num_ctx 8192)"]
  GEN --> VER["VerificationModule<br/>lexical support first; LLM judge only when uncertain"]
  GEN --> VIZ["VisualizerAgent<br/>series → fixed renderer, else gated LLM detection"]
  SER --> VIZ
  VER --> OUT["Final response<br/>answer · sources · chart · warning · trace"]
  VIZ --> OUT
  OUT --> CACHE["ResponseCache (Redis)<br/>only verified, non-degraded answers"]
  OUT --> API

  subgraph ACT["Actions  (actions/*)  — draft → human approves → execute"]
    direction TB
    EXT["ActionExtractor<br/>recipient from contacts allowlist · file path + content · edit-aware"]
    REG["ActionRegistry<br/>pending drafts · audit log"]
    WS["WorkspaceAgent<br/>confined to praxis-workspace/"]
    TG["TelegramClient"]
    GMAIL["GmailClient<br/>lazy retry"]
    EXT --> REG --> WS
    REG --> TG
    REG --> GMAIL
  end

  subgraph ING["Ingestion  (ingestion/*)"]
    direction TB
    PARSE["DocumentParser<br/>PyPDF text layer → quality score per page"]
    OCR["pdf_ocr: render (pypdfium2) → EasyOCR<br/>for absent/garbage pages"]
    SPLIT["RecursiveCharacterTextSplitter 1000/200<br/>metadata: source, page, extraction"]
    PARSE --> OCR --> SPLIT
  end
  API -->|/api/ingest| ING --> HYB

  subgraph EVAL["Evaluation  (eval/*)  — not on the request path"]
    direction TB
    E1["run_eval: routing · retrieval · answers"]
    E2["ablation: model alone → dense → +rerank → hybrid → +rerank → end-to-end"]
    E3["failure_analysis · human_eval · BASELINE.md"]
  end
```

## Where the time goes (one traced knowledge-base request, no cache)

| stage | seconds |
|---|---|
| router (rules / probe / classifier) | 0.0 – 1.2 |
| hybrid retrieval | 0.3 |
| cross-encoder rerank | 0.8 – 1.2 |
| generation (~150 words, 3B model) | 6 – 9 |
| verification (lexical first; LLM only when uncertain) | 0 – 3 |
| **total** | **~8 – 12**; cache hit 0.004 |

## Design rules the code follows

1. **A small model never transcribes numbers.** Chart figures come from parsed tables; the
   model only writes prose. (Measured: it hands full-year totals to quarters otherwise.)
2. **Deterministic before probabilistic.** Routing rules, the knowledge-base probe, file-target
   resolution and period-query phrasing are code; the classifier is the fallback.
3. **Every outbound action is a draft** until a human approves it, and every attempt is audited.
4. **Degraded is labelled, not hidden.** A rate-limited search says so in the timeline and is
   never cached.
5. **Every claim has a number** in `backend/eval/BASELINE.md`, reproducible with `eval/run_all.sh`.
