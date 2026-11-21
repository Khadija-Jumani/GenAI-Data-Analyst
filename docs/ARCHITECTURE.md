# 🏛️ GenAI Data Analyst — Architecture & System Design

This document details the architectural design, component responsibilities, data flow, and design patterns powering the **GenAI Data Analyst** platform.

---

## 1. Architectural Overview

The platform is designed as a modular, decoupled full-stack architecture:

```
                            +-----------------------------------+
                            |         Next.js Frontend          |
                            |  (React 18 / TypeScript / Tailwind) |
                            +-----------------+-----------------+
                                              |
                                     HTTP REST / JSON
                                              |
                                              v
                            +-----------------------------------+
                            |          FastAPI Backend          |
                            |  (Python 3.11+ / Uvicorn ASGI)    |
                            +-----------------+-----------------+
                                              |
                  +---------------------------+---------------------------+
                  |                           |                           |
                  v                           v                           v
     +-------------------------+ +-------------------------+ +-------------------------+
     |   Data Engine (EDA)     | | Visualization Engine    | | Generative AI Engine    |
     | (Pandas / NumPy / SciPy)| | (Plotly / PlotlyJSON)   | | (Google Gemini SDK)     |
     +-------------------------+ +-------------------------+ +-------------------------+
                  |                           |                           |
                  +---------------------------+---------------------------+
                                              |
                                              v
                            +-----------------------------------+
                            |  Autonomous Agent & Orchestrator  |
                            | (Tool Execution / Health Scoring) |
                            +-----------------------------------+
```

---

## 2. Core Subsystems

### A. Data Ingestion & Session Management (`app.services.dataset_service`)
- Validates file formats (`.csv`, `.xlsx`, `.xls`) and size limits.
- Persists data to secured local storage and registers session mappings in an in-memory session registry.
- Provides standard DataFrame accessors with type conversions and memory optimization.

### B. Statistical EDA Engine (`app.services.eda_service`)
- Computes comprehensive descriptive and higher-order statistics:
  - Missingness metrics and missing pattern flags.
  - Deduplication analysis with exact duplicate identification.
  - Spread, skewness, kurtosis, and quantile boundaries (`1%` to `99%`).
  - Tukey's IQR outlier detection ($Q_1 - 1.5 \times \text{IQR}$, $Q_3 + 1.5 \times \text{IQR}$).
  - Pearson & Spearman correlation matrices and collinearity detection.

### C. Visualization Engine (`app.services.visualization_service`)
- Leverages Plotly Express and Graph Objects.
- Serializes charts into standard Plotly JSON specifications (`data`, `layout`, `config`) using `PlotlyJSONEncoder`.
- Allows the frontend to render high-performance, responsive charts natively without server-side image generation or rendering bottlenecks.

### D. Generative AI Subsystem (`app.services.ai_service`)
- Connects to Google Gemini API (`gemini-1.5-flash`).
- Compiles rich structured statistical context from the EDA engine before prompting, avoiding hallucination and grounding all narrative answers in mathematical reality.
- Features an automated heuristic fallback engine to guarantee zero downtime when operating offline or during API quota limits.

### E. Conversational Assistant (`app.services.assistant_service`)
- Manages multi-turn conversation memory per dataset session.
- Incorporates natural-language intent recognition: automatically detects visualization requests and pairs conversational answers with newly generated interactive Plotly charts.
- Generates dynamic follow-up prompt pills.

### F. Autonomous Analytics Agent (`app.services.agent_service`)
- Multi-step tool-calling agent that plans and executes investigative workflows.
- Evaluates an objective **Data Health Score (0–100)** factoring in missingness, duplicates, and outlier volume.
- Prescribes concrete feature engineering and modeling roadmaps.

### G. Report Generation Engine (`app.services.report_service`)
- Multi-format compilation engine exporting findings to:
  1. GitHub Flavored Markdown (`.md`).
  2. Standalone, responsive HTML with print-ready CSS (`.html`).
  3. Structured JSON dossier (`.json`).

---

## 3. Security & Design Patterns

| Pattern | Implementation |
|---|---|
| **Application Factory** | `create_application()` in `app.main` ensures testability and clean lifecycle management. |
| **Pydantic V2 Type Safety** | Every endpoint strictly enforces request and response data contracts via Pydantic schemas. |
| **Layered Decoupling** | API routes (`routes/`) only handle HTTP contracts; business logic resides strictly in `services/`. |
| **Graceful Degradation** | Missing Gemini API keys gracefully fall back to local rule-based heuristic generation. |
| **CORS Middleware** | Configurable origin controls restricting API access to sanctioned frontend domains. |
