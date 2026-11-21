# 📚 GenAI Data Analyst — API Documentation

Welcome to the **GenAI Data Analyst** REST API reference.

Interactive API exploration is available locally at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI Schema**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## Base Configuration

- **Default Base URL:** `http://localhost:8000`
- **API Prefix:** `/api/v1`
- **Content-Type:** `application/json` (except file uploads which use `multipart/form-data`)

---

## 1. Core & Health Check

### `GET /`
Verifies backend operational status.
- **Response (200 OK):**
```json
{
  "status": "ok",
  "app": "GenAI Data Analyst",
  "version": "0.1.0"
}
```

---

## 2. Dataset Ingestion (`/api/v1/datasets`)

### `POST /api/v1/datasets/upload`
Upload a CSV or Excel (`.xlsx`, `.xls`) file to start an analytical session.
- **Request:** `multipart/form-data` with field `file`.
- **Response (201 Created):**
```json
{
  "dataset_id": "eb5121a10d8148f1b71e25118365dccb",
  "filename": "healthcare_patients.csv",
  "file_size_bytes": 1284,
  "rows": 25,
  "columns": 11,
  "extension": ".csv",
  "message": "Dataset uploaded successfully."
}
```

### `GET /api/v1/datasets/{id}/preview?rows=10`
Inspect preview rows from the uploaded dataset.

### `GET /api/v1/datasets/{id}/structure`
Detailed per-column metadata including Pandas dtype, category (`numeric`, `categorical`, `datetime`), null counts, and unique value counts.

### `GET /api/v1/datasets/{id}/summary`
Fast descriptive statistics for numerical columns and frequency statistics for categorical columns.

---

## 3. Automated Exploratory Data Analysis (`/api/v1/eda`)

### `GET /api/v1/eda/{id}/missing`
Comprehensive missing value analysis per column and across the entire dataset.

### `GET /api/v1/eda/{id}/duplicates`
Detects duplicate records, calculates duplicate percentages, and returns sample duplicate rows.

### `GET /api/v1/eda/{id}/numeric`
Advanced statistical profiling: mean, std, variance, range, skewness, kurtosis, quantiles (`1%`, `5%`, `25%`, `50%`, `75%`, `95%`, `99%`), and Tukey IQR outliers.

### `GET /api/v1/eda/{id}/categorical`
Cardinality assessment (`Low`, `Medium`, `High`), ranked frequencies, and rare category flags (<1% share).

### `GET /api/v1/eda/{id}/correlations`
Pearson and Spearman correlation matrices and top positive/negative feature pairs.

### `GET /api/v1/eda/{id}/distributions`
Binned histogram distributions for continuous columns and frequency tables for categoricals.

### `GET /api/v1/eda/{id}/comprehensive`
Consolidated full EDA report compiling all modules alongside rule-based analytical insights.

---

## 4. Interactive Data Visualization (`/api/v1/visualizations`)

Returns Plotly JSON chart specifications (`data`, `layout`, `config`) ready for direct rendering in React/Next.js via `react-plotly.js`.

### `GET /api/v1/visualizations/{id}/heatmap?method=pearson`
Annotated correlation heatmap with diverging `RdBu_r` color palette.

### `GET /api/v1/visualizations/{id}/distribution?column={name}`
Histogram with marginal box plot (numeric) or category frequency bar chart (categorical).

### `GET /api/v1/visualizations/{id}/box?y_column={name}&x_column={group}`
Box-and-whisker plot highlighting medians, IQR fences, and statistical outliers with optional grouping.

### `GET /api/v1/visualizations/{id}/scatter?x_column={x}&y_column={y}&color_column={color}&add_trendline=false`
Bivariate scatter plot with optional color grouping and OLS linear trendlines.

### `GET /api/v1/visualizations/{id}/bar?x_column={x}&y_column={y}&aggregation=mean`
Aggregated bar charts (`mean`, `sum`) or raw category counts.

### `GET /api/v1/visualizations/{id}/dashboard`
Curated bundle of starter charts for immediate dashboard rendering.

---

## 5. Google Gemini AI Insights (`/api/v1/ai`)

### `GET /api/v1/ai/{id}/insights`
Executive narrative summary, top statistical findings, data hygiene risks, and modeling recommendations.

### `GET /api/v1/ai/{id}/explain-feature?column={name}`
Deep statistical spread analysis and domain-specific interpretation (e.g. clinical relevance).

### `GET /api/v1/ai/{id}/explain-correlation?feature_a={a}&feature_b={b}`
Contextual explanation of bivariate association, confounding variables, and causation cautions.

### `POST /api/v1/ai/{id}/ask`
Context-aware dataset Q&A responding to natural language questions based on dataset statistics.
- **Request Body:**
```json
{
  "question": "What is the average cholesterol level across diabetic patients?"
}
```

---

## 6. Natural-Language Assistant (`/api/v1/assistant`)

### `POST /api/v1/assistant/{id}/chat`
Multi-turn conversational assistant with automated chart intent detection. When a user asks for a chart (*"plot box plot of cholesterol by diabetes_risk"*), the assistant automatically generates and attaches the interactive Plotly specification!
- **Request Body:**
```json
{
  "session_id": "optional-session-id",
  "message": "Show me a box plot of cholesterol by diabetes_risk."
}
```

### `GET /api/v1/assistant/{id}/history?session_id={id}`
Retrieves complete conversation dialogue history.

### `DELETE /api/v1/assistant/{id}/history?session_id={id}`
Resets conversational memory for the session.

---

## 7. Autonomous Analytics Agent (`/api/v1/agent`)

### `GET /api/v1/agent/{id}/workflows`
Lists available autonomous analytical investigation workflows.

### `POST /api/v1/agent/{id}/workflow`
Executes an autonomous multi-step tool-assisted analytical workflow:
- `comprehensive_audit`: Full multi-step audit of data health, outliers, and collinearity.
- `feature_engineering`: Prescribes scaling, transformations, and encoding strategies.
- `ml_readiness`: Evaluates sample size, target class balance, and matches candidate ML models.

---

## 8. Report Generation & Export (`/api/v1/reports`)

### `GET /api/v1/reports/{id}/summary`
Report overview, overall data health score, and export links.

### `GET /api/v1/reports/{id}/markdown?download=true`
Exports the complete analytical report in GitHub Flavored Markdown (`.md`).

### `GET /api/v1/reports/{id}/html?download=true`
Exports a self-contained, beautifully styled HTML document with print-ready CSS (`.html`).

### `GET /api/v1/reports/{id}/json?download=true`
Exports the complete machine-readable analytical dossier in JSON (`.json`).
