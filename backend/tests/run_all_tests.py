"""
backend/tests/run_all_tests.py
------------------------------
Unified Master Test Runner for GenAI Data Analyst.

Executes all analytical test suites:
- test_eda.py (Phase 2 & 3: Ingestion & Statistical EDA)
- test_visualization.py (Phase 4: Plotly Interactive Visualizations)
- test_ai.py (Phase 5: Google Gemini AI Insights)
- test_assistant.py (Phase 6: Conversational Assistant & Chart Detection)
- test_agent.py (Phase 7: Autonomous Agent Workflows & Health Scoring)
- test_report.py (Phase 8: Multi-format Report Generation & Export)
"""

import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from tests.test_agent import test_agent_workflows_pipeline
from tests.test_ai import test_ai_pipeline
from tests.test_assistant import test_assistant_pipeline
from tests.test_eda import test_full_eda_pipeline
from tests.test_report import test_report_pipeline
from tests.test_visualization import test_visualization_pipeline


def run_all():
    suites = [
        ("Phase 3: Automated EDA Engine", test_full_eda_pipeline),
        ("Phase 4: Interactive Visualizations", test_visualization_pipeline),
        ("Phase 5: Google Gemini AI Insights", test_ai_pipeline),
        ("Phase 6: Conversational Assistant", test_assistant_pipeline),
        ("Phase 7: Autonomous Agent Workflows", test_agent_workflows_pipeline),
        ("Phase 8: Report Generation & Export", test_report_pipeline),
    ]

    print("\n" + "=" * 65)
    print("   GENAI DATA ANALYST — UNIFIED SYSTEM TEST RUNNER")
    print("=" * 65 + "\n")

    passed_count = 0
    failed_count = 0
    results = []

    total_start = time.time()

    for name, runner in suites:
        print(f"\n>>> Running: {name} ...")
        t0 = time.time()
        try:
            runner()
            duration = round(time.time() - t0, 3)
            passed_count += 1
            results.append((name, "PASSED", f"{duration}s"))
        except Exception as exc:
            duration = round(time.time() - t0, 3)
            failed_count += 1
            results.append((name, f"FAILED: {exc}", f"{duration}s"))

    total_duration = round(time.time() - total_start, 3)

    print("\n" + "=" * 65)
    print("                 TEST EXECUTION SUMMARY")
    print("=" * 65)
    for name, status, dur in results:
        indicator = "✅" if status == "PASSED" else "❌"
        print(f" {indicator}  {name:<40} {status:<10} ({dur})")
    print("-" * 65)
    print(f" Total Tests Run: {len(suites)} | Passed: {passed_count} | Failed: {failed_count} | Duration: {total_duration}s")
    print("=" * 65 + "\n")

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
