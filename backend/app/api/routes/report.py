"""
backend/app/api/routes/report.py
--------------------------------
FastAPI routes for Phase 8: Report Generation and Export.

Endpoints:
- GET /api/v1/reports/{dataset_id}/summary   Report overview and export URLs
- GET /api/v1/reports/{dataset_id}/markdown  Export report in Markdown (.md)
- GET /api/v1/reports/{dataset_id}/html      Export standalone styled HTML (.html)
- GET /api/v1/reports/{dataset_id}/json      Export analytical dossier in JSON (.json)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import HTMLResponse, PlainTextResponse

from app.schemas.report import ReportSummaryResponse
from app.services import report_service

router = APIRouter(prefix="/reports", tags=["Analytical Reports & Export"])


@router.get(
    "/{dataset_id}/summary",
    response_model=ReportSummaryResponse,
    summary="Report overview & available downloads",
    description="Retrieve high-level report metadata, health score, and export URLs.",
)
def get_report_summary(dataset_id: str) -> ReportSummaryResponse:
    try:
        return report_service.get_report_summary(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/markdown",
    response_class=PlainTextResponse,
    summary="Export analytical report in Markdown (.md)",
    description="Download or stream the complete analytical findings as a GitHub Flavored Markdown document.",
)
def get_markdown_report(
    dataset_id: str,
    download: bool = Query(default=False, description="Set true to force browser download as file attachment."),
) -> Response:
    try:
        md_content = report_service.generate_markdown_report(dataset_id)
        headers = {}
        if download:
            headers["Content-Disposition"] = f"attachment; filename=report_{dataset_id[:8]}.md"
        return PlainTextResponse(content=md_content, media_type="text/markdown", headers=headers)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/html",
    response_class=HTMLResponse,
    summary="Export analytical report in standalone HTML (.html)",
    description="Download or preview the self-contained styled executive report with print-ready CSS.",
)
def get_html_report(
    dataset_id: str,
    download: bool = Query(default=False, description="Set true to force browser download as file attachment."),
) -> Response:
    try:
        html_content = report_service.generate_html_report(dataset_id)
        headers = {}
        if download:
            headers["Content-Disposition"] = f"attachment; filename=report_{dataset_id[:8]}.html"
        return HTMLResponse(content=html_content, headers=headers)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/{dataset_id}/json",
    summary="Export analytical dossier in JSON (.json)",
    description="Download complete machine-readable audit and statistics dossier.",
)
def get_json_report(
    dataset_id: str,
    download: bool = Query(default=False, description="Set true to force browser download as file attachment."),
) -> Any:
    try:
        data = report_service.generate_json_dossier(dataset_id)
        headers = {}
        if download:
            headers["Content-Disposition"] = f"attachment; filename=dossier_{dataset_id[:8]}.json"
        return Response(
            content=data if isinstance(data, str) else None,
            media_type="application/json",
            headers=headers,
        ) if isinstance(data, str) else data
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
