"""
backend/tests/test_agent.py
---------------------------
Automated test suite verifying Phase 7 Agentic AI and Autonomous Analytical Workflows.
"""

import sys
from pathlib import Path
from starlette.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.main import app

client = TestClient(app)


def test_agent_workflows_pipeline():
    sample_csv = Path(__file__).resolve().parent.parent.parent / "datasets" / "sample" / "healthcare_patients.csv"
    assert sample_csv.exists(), f"Sample dataset not found at {sample_csv}"

    # 1. Upload sample dataset
    with open(sample_csv, "rb") as f:
        res = client.post("/api/v1/datasets/upload", files={"file": ("healthcare_patients.csv", f, "text/csv")})
    assert res.status_code == 201
    dataset_id = res.json()["dataset_id"]
    print(f"[PASSED] Uploaded dataset ID: {dataset_id}")

    # 2. List Available Workflows
    res = client.get(f"/api/v1/agent/{dataset_id}/workflows")
    assert res.status_code == 200, res.text
    wf_list = res.json()["workflows"]
    assert len(wf_list) >= 3
    print(f"[PASSED] Listed {len(wf_list)} available agent workflows.")

    # 3. Workflow 1: Comprehensive Diagnostic Audit
    res = client.post(
        f"/api/v1/agent/{dataset_id}/workflow",
        json={"workflow_type": "comprehensive_audit"},
    )
    assert res.status_code == 200, res.text
    audit = res.json()
    assert audit["workflow_type"] == "comprehensive_audit"
    assert audit["overall_health_score"] > 0
    assert len(audit["execution_steps"]) == 4
    assert len(audit["key_findings"]) > 0
    assert len(audit["prescribed_actions"]) > 0
    print(f"[PASSED] Comprehensive Audit executed. Health Score: {audit['overall_health_score']}/100. Steps: {len(audit['execution_steps'])}")

    # 4. Workflow 2: Feature Engineering Roadmap
    res = client.post(
        f"/api/v1/agent/{dataset_id}/workflow",
        json={"workflow_type": "feature_engineering"},
    )
    assert res.status_code == 200, res.text
    fe = res.json()
    assert fe["workflow_type"] == "feature_engineering"
    assert len(fe["execution_steps"]) == 3
    assert len(fe["prescribed_actions"]) > 0
    print(f"[PASSED] Feature Engineering Roadmap executed with {len(fe['execution_steps'])} steps.")

    # 5. Workflow 3: Predictive ML Readiness Assessment
    res = client.post(
        f"/api/v1/agent/{dataset_id}/workflow",
        json={"workflow_type": "ml_readiness", "target_column": "diabetes_risk"},
    )
    assert res.status_code == 200, res.text
    ml = res.json()
    assert ml["workflow_type"] == "ml_readiness"
    assert len(ml["recommended_models"]) > 0
    print(f"[PASSED] ML Readiness Assessment executed. Target: diabetes_risk. Models: {', '.join(ml['recommended_models'][:2])}")

    print("\n==================================================")
    print("ALL PHASE 7 AGENTIC WORKFLOW TESTS PASSED 100%!")
    print("==================================================")


if __name__ == "__main__":
    test_agent_workflows_pipeline()
