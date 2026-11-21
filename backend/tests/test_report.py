"""
backend/tests/test_report.py
----------------------------
Automated test suite verifying Phase 8 Report Generation and Export endpoints.
"""

import sys
from pathlib import Path
from starlette.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.main import app

client = TestClient(app)


def test_report_pipeline():
    sample_csv = Path(__file__).resolve().parent.parent.parent / "datasets" / "sample" / "healthcare_patients.csv"
    assert sample_csv.exists(), f"Sample dataset not found at {sample_csv}"

    # 1. Upload sample dataset
    with open(sample_csv, "rb") as f:
        res = client.post("/api/v1/datasets/upload", files={"file": ("healthcare_patients.csv", f, "text/csv")})
    assert res.status_code == 201
    dataset_id = res.json()["dataset_id"]
    print(f"[PASSED] Uploaded dataset ID: {dataset_id}")

    # 2. Report Summary
    res = client.get(f"/api/v1/reports/{dataset_id}/summary")
    assert res.status_code == 200, res.text
    summary = res.json()
    assert summary["dataset_id"] == dataset_id
    assert summary["overall_health_score"] > 0
    assert len(summary["sections_included"]) >= 5
    assert "markdown" in summary["download_urls"]
    assert "html" in summary["download_urls"]
    assert "json" in summary["download_urls"]
    print(f"[PASSED] Report Summary verified. Health score: {summary['overall_health_score']}/100.")

    # 3. Markdown Export
    res = client.get(f"/api/v1/reports/{dataset_id}/markdown?download=true")
    assert res.status_code == 200, res.text
    assert "text/markdown" in res.headers["content-type"]
    assert "attachment" in res.headers["content-disposition"]
    md_text = res.text
    assert "# 📊 GenAI Data Analyst — Analytical Report" in md_text
    assert "Executive Summary" in md_text
    assert "Numerical Features Analysis" in md_text
    print(f"[PASSED] Markdown report generated ({len(md_text)} chars).")

    # 4. HTML Export
    res = client.get(f"/api/v1/reports/{dataset_id}/html?download=true")
    assert res.status_code == 200, res.text
    assert "text/html" in res.headers["content-type"]
    assert "attachment" in res.headers["content-disposition"]
    html_text = res.text
    assert "<!DOCTYPE html>" in html_text
    assert "GenAI Data Analyst Report" in html_text
    assert "stat-card" in html_text
    print(f"[PASSED] Standalone HTML report generated ({len(html_text)} chars).")

    # 5. JSON Dossier Export
    res = client.get(f"/api/v1/reports/{dataset_id}/json")
    assert res.status_code == 200, res.text
    dossier = res.json()
    assert "report_title" in dossier
    assert "data_health_score" in dossier
    assert "dataset_shape" in dossier
    assert "executive_insights" in dossier
    assert "eda_report" in dossier
    print(f"[PASSED] Machine-readable JSON dossier generated with {len(dossier.keys())} top-level sections.")

    print("\n=======================================================")
    print("ALL PHASE 8 REPORT GENERATION & EXPORT TESTS PASSED 100%!")
    print("=======================================================")


if __name__ == "__main__":
    test_report_pipeline()
