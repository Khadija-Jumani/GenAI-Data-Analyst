"""
backend/tests/test_eda.py
-------------------------
Automated test suite verifying Phase 3 Exploratory Data Analysis endpoints.
"""

import sys
from pathlib import Path
from starlette.testclient import TestClient

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.main import app

client = TestClient(app)


def test_full_eda_pipeline():
    sample_csv = Path(__file__).resolve().parent.parent.parent / "datasets" / "sample" / "healthcare_patients.csv"
    assert sample_csv.exists(), f"Sample dataset not found at {sample_csv}"

    # 1. Upload
    with open(sample_csv, "rb") as f:
        res = client.post("/api/v1/datasets/upload", files={"file": ("healthcare_patients.csv", f, "text/csv")})
    assert res.status_code == 201, res.text
    dataset_id = res.json()["dataset_id"]
    assert dataset_id is not None
    print(f"[PASSED] Dataset uploaded. ID: {dataset_id}")

    # 2. Missing Value Analysis
    res = client.get(f"/api/v1/eda/{dataset_id}/missing")
    assert res.status_code == 200, res.text
    missing = res.json()
    assert "total_missing_cells" in missing
    assert "columns" in missing
    print(f"[PASSED] Missing values analysis: {missing['total_missing_cells']} missing cells.")

    # 3. Duplicate Analysis
    res = client.get(f"/api/v1/eda/{dataset_id}/duplicates")
    assert res.status_code == 200, res.text
    dups = res.json()
    assert dups["has_duplicates"] is False
    print(f"[PASSED] Duplicate analysis: {dups['duplicate_rows_count']} duplicates found.")

    # 4. Detailed Numeric Features
    res = client.get(f"/api/v1/eda/{dataset_id}/numeric")
    assert res.status_code == 200, res.text
    numerics = res.json()
    assert len(numerics) > 0
    first_col = numerics[0]
    assert "skewness" in first_col
    assert "kurtosis" in first_col
    assert "quantiles" in first_col
    assert "outliers" in first_col
    print(f"[PASSED] Numeric features analysis: {len(numerics)} columns analyzed.")

    # 5. Categorical Features
    res = client.get(f"/api/v1/eda/{dataset_id}/categorical")
    assert res.status_code == 200, res.text
    categoricals = res.json()
    assert len(categoricals) > 0
    assert "cardinality" in categoricals[0]
    assert "top_categories" in categoricals[0]
    print(f"[PASSED] Categorical features analysis: {len(categoricals)} columns analyzed.")

    # 6. Correlations
    res = client.get(f"/api/v1/eda/{dataset_id}/correlations")
    assert res.status_code == 200, res.text
    corrs = res.json()
    assert "pearson_matrix" in corrs
    assert "spearman_matrix" in corrs
    assert "top_positive_correlations" in corrs
    print(f"[PASSED] Correlation matrices computed. Top positive pairs: {len(corrs['top_positive_correlations'])}")

    # 7. Distributions
    res = client.get(f"/api/v1/eda/{dataset_id}/distributions")
    assert res.status_code == 200, res.text
    dist = res.json()
    assert len(dist["numeric_histograms"]) > 0
    print(f"[PASSED] Distribution histograms computed: {len(dist['numeric_histograms'])} numeric histograms.")

    # 8. Comprehensive Report
    res = client.get(f"/api/v1/eda/{dataset_id}/comprehensive")
    assert res.status_code == 200, res.text
    comp = res.json()
    assert "key_insights" in comp
    assert len(comp["key_insights"]) > 0
    print(f"[PASSED] Comprehensive report generated with {len(comp['key_insights'])} insights:")
    for insight in comp["key_insights"]:
        print(f"   {insight}")

    print("\n==========================================")
    print("ALL PHASE 3 EDA TESTS PASSED SUCCESSFULLY!")
    print("==========================================")


if __name__ == "__main__":
    test_full_eda_pipeline()
