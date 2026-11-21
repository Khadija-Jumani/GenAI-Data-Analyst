"""
backend/tests/test_ai.py
------------------------
Automated test suite verifying Phase 5 Generative AI endpoints.
"""

import sys
from pathlib import Path
from starlette.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.main import app

client = TestClient(app)


def test_ai_pipeline():
    sample_csv = Path(__file__).resolve().parent.parent.parent / "datasets" / "sample" / "healthcare_patients.csv"
    assert sample_csv.exists(), f"Sample dataset not found at {sample_csv}"

    # 1. Upload sample dataset
    with open(sample_csv, "rb") as f:
        res = client.post("/api/v1/datasets/upload", files={"file": ("healthcare_patients.csv", f, "text/csv")})
    assert res.status_code == 201
    dataset_id = res.json()["dataset_id"]
    print(f"[PASSED] Uploaded dataset ID: {dataset_id}")

    # 2. Executive Insights
    res = client.get(f"/api/v1/ai/{dataset_id}/insights")
    assert res.status_code == 200, res.text
    insights = res.json()
    assert "executive_summary" in insights
    assert len(insights["key_findings"]) > 0
    assert len(insights["anomalies_and_risks"]) > 0
    assert len(insights["actionable_recommendations"]) > 0
    print(f"[PASSED] Executive insights generated using: {insights['model_used']}")
    print(f"   Summary: {insights['executive_summary'][:100]}...")

    # 3. Feature Explanation
    res = client.get(f"/api/v1/ai/{dataset_id}/explain-feature?column=cholesterol")
    assert res.status_code == 200, res.text
    feat_exp = res.json()
    assert feat_exp["column"] == "cholesterol"
    assert "statistical_summary" in feat_exp
    assert "domain_significance" in feat_exp
    print(f"[PASSED] Feature explanation generated for cholesterol.")

    # 4. Correlation Explanation
    res = client.get(f"/api/v1/ai/{dataset_id}/explain-correlation?feature_a=systolic_bp&feature_b=cholesterol")
    assert res.status_code == 200, res.text
    corr_exp = res.json()
    assert corr_exp["feature_a"] == "systolic_bp"
    assert "relationship_narrative" in corr_exp
    assert "causation_caution" in corr_exp
    print(f"[PASSED] Correlation explanation generated (r={corr_exp['correlation_score']}).")

    # 5. Natural Language Dataset Q&A
    res = client.post(
        f"/api/v1/ai/{dataset_id}/ask",
        json={"question": "What is the average cholesterol and are there any missing values?"},
    )
    assert res.status_code == 200, res.text
    qa = res.json()
    assert "answer" in qa
    assert qa["confidence"] in ["High", "Medium", "Low"]
    print(f"[PASSED] Natural language Q&A response: {qa['answer']}")

    print("\n=============================================")
    print("ALL PHASE 5 GENERATIVE AI TESTS PASSED 100%!")
    print("=============================================")


if __name__ == "__main__":
    test_ai_pipeline()
