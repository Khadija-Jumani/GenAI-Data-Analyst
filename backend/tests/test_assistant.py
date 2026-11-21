"""
backend/tests/test_assistant.py
-------------------------------
Automated test suite verifying Phase 6 Natural-Language Assistant endpoints.
"""

import sys
from pathlib import Path
from starlette.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.main import app

client = TestClient(app)


def test_assistant_pipeline():
    sample_csv = Path(__file__).resolve().parent.parent.parent / "datasets" / "sample" / "healthcare_patients.csv"
    assert sample_csv.exists(), f"Sample dataset not found at {sample_csv}"

    # 1. Upload sample dataset
    with open(sample_csv, "rb") as f:
        res = client.post("/api/v1/datasets/upload", files={"file": ("healthcare_patients.csv", f, "text/csv")})
    assert res.status_code == 201
    dataset_id = res.json()["dataset_id"]
    print(f"[PASSED] Uploaded dataset ID: {dataset_id}")

    # 2. Turn 1: General question
    res = client.post(
        f"/api/v1/assistant/{dataset_id}/chat",
        json={"message": "Can you summarize the patient demographics in this dataset?"},
    )
    assert res.status_code == 200, res.text
    turn1 = res.json()
    session_id = turn1["session_id"]
    assert session_id is not None
    assert turn1["reply"]["role"] == "assistant"
    assert len(turn1["reply"]["content"]) > 0
    assert len(turn1["suggested_followups"]) > 0
    print(f"[PASSED] Turn 1 complete (Session ID: {session_id[:8]}...).")

    # 3. Turn 2: Natural-language chart request (Box plot)
    res = client.post(
        f"/api/v1/assistant/{dataset_id}/chat",
        json={
            "session_id": session_id,
            "message": "Please show me a box plot of cholesterol by diabetes_risk.",
        },
    )
    assert res.status_code == 200, res.text
    turn2 = res.json()
    assert turn2["session_id"] == session_id
    assert turn2["reply"]["chart_attachment"] is not None
    assert turn2["reply"]["chart_attachment"]["chart_type"] == "box"
    print(f"[PASSED] Turn 2 chart generation: {turn2['reply']['chart_attachment']['title']}.")

    # 4. Turn 3: Natural-language scatter plot
    res = client.post(
        f"/api/v1/assistant/{dataset_id}/chat",
        json={
            "session_id": session_id,
            "message": "Create a scatter plot of systolic_bp vs cholesterol with smoking_status color.",
        },
    )
    assert res.status_code == 200, res.text
    turn3 = res.json()
    assert turn3["reply"]["chart_attachment"] is not None
    assert turn3["reply"]["chart_attachment"]["chart_type"] == "scatter"
    print(f"[PASSED] Turn 3 scatter plot generated.")

    # 5. Fetch Session History
    res = client.get(f"/api/v1/assistant/{dataset_id}/history?session_id={session_id}")
    assert res.status_code == 200, res.text
    history = res.json()
    assert history["message_count"] == 6  # 3 user messages + 3 assistant responses
    assert len(history["messages"]) == 6
    print(f"[PASSED] Session history verified with {history['message_count']} messages.")

    # 6. Clear Session History
    res = client.delete(f"/api/v1/assistant/{dataset_id}/history?session_id={session_id}")
    assert res.status_code == 200, res.text

    # Re-check history is empty
    res = client.get(f"/api/v1/assistant/{dataset_id}/history?session_id={session_id}")
    assert res.status_code == 200
    assert res.json()["message_count"] == 0
    print(f"[PASSED] Session history successfully cleared.")

    print("\n=======================================================")
    print("ALL PHASE 6 NATURAL-LANGUAGE ASSISTANT TESTS PASSED 100%!")
    print("=======================================================")


if __name__ == "__main__":
    test_assistant_pipeline()
