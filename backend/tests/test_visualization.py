"""
backend/tests/test_visualization.py
-----------------------------------
Automated test suite verifying Phase 4 Interactive Data Visualization endpoints.
"""

import sys
from pathlib import Path
from starlette.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.main import app

client = TestClient(app)


def test_visualization_pipeline():
    sample_csv = Path(__file__).resolve().parent.parent.parent / "datasets" / "sample" / "healthcare_patients.csv"
    assert sample_csv.exists(), f"Sample dataset not found at {sample_csv}"

    # 1. Upload sample dataset
    with open(sample_csv, "rb") as f:
        res = client.post("/api/v1/datasets/upload", files={"file": ("healthcare_patients.csv", f, "text/csv")})
    assert res.status_code == 201
    dataset_id = res.json()["dataset_id"]
    print(f"[PASSED] Uploaded dataset ID: {dataset_id}")

    # 2. Correlation Heatmap
    res = client.get(f"/api/v1/visualizations/{dataset_id}/heatmap?method=pearson")
    assert res.status_code == 200, res.text
    heatmap = res.json()
    assert heatmap["chart_type"] == "heatmap"
    assert "data" in heatmap["spec"]
    assert "layout" in heatmap["spec"]
    print(f"[PASSED] Heatmap generated with {len(heatmap['spec']['data'])} trace(s).")

    # 3. Numeric Distribution (fasting_blood_sugar)
    res = client.get(f"/api/v1/visualizations/{dataset_id}/distribution?column=fasting_blood_sugar")
    assert res.status_code == 200, res.text
    num_dist = res.json()
    assert num_dist["chart_type"] == "histogram"
    assert len(num_dist["spec"]["data"]) > 0
    print(f"[PASSED] Numeric distribution plot generated for fasting_blood_sugar.")

    # 4. Categorical Distribution (diabetes_risk)
    res = client.get(f"/api/v1/visualizations/{dataset_id}/distribution?column=diabetes_risk")
    assert res.status_code == 200, res.text
    cat_dist = res.json()
    assert cat_dist["chart_type"] == "bar"
    assert len(cat_dist["spec"]["data"]) > 0
    print(f"[PASSED] Categorical distribution bar chart generated for diabetes_risk.")

    # 5. Box Plot (cholesterol by diabetes_risk)
    res = client.get(f"/api/v1/visualizations/{dataset_id}/box?y_column=cholesterol&x_column=diabetes_risk")
    assert res.status_code == 200, res.text
    box_plot = res.json()
    assert box_plot["chart_type"] == "box"
    assert len(box_plot["spec"]["data"]) > 0
    print(f"[PASSED] Box plot generated for cholesterol grouped by diabetes_risk.")

    # 6. Scatter Plot (systolic_bp vs cholesterol with smoking_status color)
    res = client.get(f"/api/v1/visualizations/{dataset_id}/scatter?x_column=systolic_bp&y_column=cholesterol&color_column=smoking_status")
    assert res.status_code == 200, res.text
    scatter = res.json()
    assert scatter["chart_type"] == "scatter"
    assert len(scatter["spec"]["data"]) > 0
    print(f"[PASSED] Scatter plot generated with group color.")

    # 7. Aggregated Bar Chart (mean cholesterol by physical_activity)
    res = client.get(f"/api/v1/visualizations/{dataset_id}/bar?x_column=physical_activity&y_column=cholesterol&aggregation=mean")
    assert res.status_code == 200, res.text
    bar_chart = res.json()
    assert bar_chart["chart_type"] == "bar"
    assert len(bar_chart["spec"]["data"]) > 0
    print(f"[PASSED] Aggregated bar chart generated.")

    # 8. Curated Dashboard Suite
    res = client.get(f"/api/v1/visualizations/{dataset_id}/dashboard")
    assert res.status_code == 200, res.text
    dashboard = res.json()
    assert len(dashboard["charts"]) >= 3
    print(f"[PASSED] Dashboard bundle generated with {len(dashboard['charts'])} starter charts:")
    for c in dashboard["charts"]:
        print(f"   - {c['chart_type'].upper()}: {c['title']}")

    print("\n===================================================")
    print("ALL PHASE 4 VISUALIZATION TESTS PASSED SUCCESSFULLY!")
    print("===================================================")


if __name__ == "__main__":
    test_visualization_pipeline()
