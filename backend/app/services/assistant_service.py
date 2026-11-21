"""
backend/app/services/assistant_service.py
-----------------------------------------
Conversational Assistant Engine for Phase 6.

Capabilities:
1. Multi-turn chat session memory per dataset.
2. User Intent Classification:
   - CHART_REQUEST: Automatically invokes visualization_service to generate Plotly charts.
   - STATISTICAL_QUERY: Queries eda_service for exact summary figures, outliers, and quantiles.
   - CORRELATION_QUERY: Analyzes pairwise relationship and causation.
   - GENERAL_DATA_QUERY: Conversational exploration and explanation.
3. Google Gemini powered conversational reasoning with seamless heuristic fallback.
4. Context-aware follow-up prompt suggestion generation.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import pandas as pd

from app.core.config import settings
from app.schemas.assistant import (
    AssistantMessageResponse,
    ChatMessage,
    ChatSessionResponse,
)
from app.schemas.visualization import PlotlyChartResponse
from app.services import ai_service, visualization_service
from app.services.dataset_service import get_dataframe
from app.services.eda_service import (
    analyze_categorical_features,
    analyze_duplicates,
    analyze_missing_values,
    analyze_numeric_features,
    compute_correlations,
)

# In-memory session registry: session_id -> { "dataset_id": str, "messages": list[ChatMessage] }
_sessions: dict[str, dict[str, Any]] = {}


def _get_or_create_session(dataset_id: str, session_id: str | None = None) -> str:
    """Retrieve existing session ID or register a new one."""
    if session_id and session_id in _sessions:
        return session_id
    new_id = uuid.uuid4().hex
    _sessions[new_id] = {
        "dataset_id": dataset_id,
        "messages": [],
    }
    return new_id


def _find_referenced_columns(df: pd.DataFrame, text: str) -> list[str]:
    """Identify column names mentioned in user prompt (case-insensitive fuzzy match)."""
    text_lower = text.lower()
    matched = []
    for col in df.columns:
        # Match exact column name or column name with underscores replaced by spaces
        col_clean = col.lower()
        col_spaced = col_clean.replace("_", " ")
        if col_clean in text_lower or col_spaced in text_lower:
            matched.append(col)
    return matched


def _detect_chart_intent(
    df: pd.DataFrame,
    text: str,
    matched_cols: list[str],
) -> tuple[bool, str, dict[str, Any]]:
    """
    Detect if the user wants an interactive chart and determine chart type and parameters.
    Returns (is_chart_request, chart_type, params_dict).
    """
    text_lower = text.lower()
    chart_keywords = ["plot", "chart", "graph", "histogram", "scatter", "box", "heatmap", "distribution", "visualize", "show me a"]

    if not any(k in text_lower for k in chart_keywords):
        return False, "", {}

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # 1. Heatmap
    if "heatmap" in text_lower or "correlation" in text_lower:
        return True, "heatmap", {"method": "pearson"}

    # 2. Box plot
    if "box" in text_lower:
        num_target = next((c for c in matched_cols if c in numeric_cols), numeric_cols[0] if numeric_cols else None)
        cat_group = next((c for c in matched_cols if c in cat_cols), None)
        if num_target:
            return True, "box", {"y_column": num_target, "x_column": cat_group}

    # 3. Scatter plot
    if "scatter" in text_lower or "versus" in text_lower or " vs " in text_lower or " vs. " in text_lower:
        nums = [c for c in matched_cols if c in numeric_cols]
        if len(nums) >= 2:
            color = next((c for c in matched_cols if c in cat_cols), None)
            return True, "scatter", {"x_column": nums[0], "y_column": nums[1], "color_column": color}
        elif len(numeric_cols) >= 2:
            return True, "scatter", {"x_column": numeric_cols[0], "y_column": numeric_cols[1], "color_column": None}

    # 4. Distribution / Histogram
    if "distribution" in text_lower or "histogram" in text_lower:
        target = matched_cols[0] if matched_cols else (numeric_cols[0] if numeric_cols else df.columns[0])
        return True, "distribution", {"column": target}

    # 5. Default bar chart or distribution if column matched
    if matched_cols:
        target = matched_cols[0]
        if target in numeric_cols:
            return True, "distribution", {"column": target}
        else:
            return True, "bar", {"x_column": target}

    return False, "", {}


def process_user_query(dataset_id: str, message: str, session_id: str | None = None) -> AssistantMessageResponse:
    """Handle a user query with multi-turn history and optional automated visualization generation."""
    df, filename = get_dataframe(dataset_id)
    sid = _get_or_create_session(dataset_id, session_id)
    session_data = _sessions[sid]

    # Record user message in history
    user_msg = ChatMessage(role="user", content=message)
    session_data["messages"].append(user_msg)

    # Context analysis
    matched_cols = _find_referenced_columns(df, message)
    is_chart, chart_type, chart_params = _detect_chart_intent(df, message, matched_cols)

    chart_attachment: PlotlyChartResponse | None = None
    if is_chart:
        try:
            if chart_type == "heatmap":
                chart_attachment = visualization_service.create_correlation_heatmap(df, **chart_params)
            elif chart_type == "box":
                chart_attachment = visualization_service.create_box_plot(df, **chart_params)
            elif chart_type == "scatter":
                chart_attachment = visualization_service.create_scatter_plot(df, **chart_params)
            elif chart_type == "distribution":
                chart_attachment = visualization_service.create_distribution_plot(df, **chart_params)
            elif chart_type == "bar":
                chart_attachment = visualization_service.create_bar_chart(df, **chart_params)
        except Exception:
            pass

    # Build assistant narrative answer
    model = ai_service._init_gemini()
    recent_history = session_data["messages"][-6:]  # Last 3 conversational turns

    history_formatted = "\n".join([f"{m.role.upper()}: {m.content}" for m in recent_history])
    context_summary = ai_service._build_context_summary(df, filename)

    content_reply = ""
    suggested_followups = []

    if model:
        chart_note = f"Note: An interactive {chart_type} chart has been generated and attached." if chart_attachment else ""
        prompt = f"""
You are GenAI Data Analyst, an elite AI data science and healthcare analytics assistant.
Respond to the user's latest query accurately and informatively based on the dataset context and conversation history.

Dataset Context:
{json.dumps(context_summary, indent=2)}

Conversation History:
{history_formatted}

{chart_note}

Format your response in GitHub Markdown. If a chart was attached, briefly interpret what the chart reveals.
Also provide 2 or 3 natural follow-up analytical questions that the user might want to explore next.

Required output format:
<RESPONSE>
Your markdown response here.
</RESPONSE>
<FOLLOWUPS>
- Follow-up question 1
- Follow-up question 2
- Follow-up question 3
</FOLLOWUPS>
"""
        try:
            resp = model.generate_content(prompt)
            raw = resp.text.strip()
            # Parse sections
            if "<RESPONSE>" in raw and "</RESPONSE>" in raw:
                content_reply = raw.split("<RESPONSE>")[1].split("</RESPONSE>")[0].strip()
            else:
                content_reply = raw

            if "<FOLLOWUPS>" in raw and "</FOLLOWUPS>" in raw:
                fu_text = raw.split("<FOLLOWUPS>")[1].split("</FOLLOWUPS>")[0].strip()
                suggested_followups = [
                    re.sub(r"^[0-9\-\*\.\s]+", "", line).strip()
                    for line in fu_text.splitlines()
                    if line.strip()
                ][:3]
        except Exception:
            pass

    # Fallback response generation
    if not content_reply:
        if chart_attachment:
            content_reply = (
                f"### 📊 Interactive Visualization Generated\n\n"
                f"I've generated an interactive **{chart_attachment.title}** for you.\n\n"
                f"- **Chart Type:** `{chart_attachment.chart_type}`\n"
                f"- **Observations:** {chart_attachment.description}\n\n"
                f"You can hover, zoom, and inspect individual data points directly on the chart."
            )
            suggested_followups = [
                f"What are the outliers in this {chart_attachment.chart_type}?",
                "Can you explain the statistical correlation between these features?",
                "What feature engineering steps should I take next?",
            ]
        else:
            qa_res = ai_service.ask_dataset(dataset_id, message)
            content_reply = f"### Analytical Response\n\n{qa_res.answer}"
            suggested_followups = [
                "Show me the distribution of this feature",
                "Are there any outliers in the numeric columns?",
                "Generate a correlation heatmap for this dataset",
            ]

    # Create assistant message
    assistant_msg = ChatMessage(
        role="assistant",
        content=content_reply,
        chart_attachment=chart_attachment,
        referenced_columns=matched_cols,
    )
    session_data["messages"].append(assistant_msg)

    return AssistantMessageResponse(
        session_id=sid,
        dataset_id=dataset_id,
        reply=assistant_msg,
        suggested_followups=suggested_followups,
    )


def get_chat_history(dataset_id: str, session_id: str | None = None) -> ChatSessionResponse:
    """Retrieve all messages in an active session."""
    df, filename = get_dataframe(dataset_id)
    sid = _get_or_create_session(dataset_id, session_id)
    session_data = _sessions[sid]

    return ChatSessionResponse(
        session_id=sid,
        dataset_id=dataset_id,
        filename=filename,
        message_count=len(session_data["messages"]),
        messages=session_data["messages"],
    )


def clear_chat_history(dataset_id: str, session_id: str | None = None) -> dict[str, str]:
    """Clear conversation history for a session."""
    if session_id and session_id in _sessions:
        _sessions[session_id]["messages"] = []
    return {"status": "cleared", "dataset_id": dataset_id}
