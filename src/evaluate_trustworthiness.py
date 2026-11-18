"""
CLI entry point to run StudyPal's Automated Trustworthiness Evaluation Suite.

Usage:
    python src/evaluate_trustworthiness.py
    python src/evaluate_trustworthiness.py --threshold 0.30
"""

from __future__ import annotations

import argparse
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from trustworthiness.evaluator import (  # noqa: E402
    run_trustworthiness_benchmark,
    save_benchmark_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run StudyPal Trustworthiness Evaluation & Red-Team Benchmark."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Minimum cosine similarity threshold for grounding/abstention (default: 0.30).",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Run only guardrails, privacy, and faithfulness verifier suites.",
    )
    args = parser.parse_args()

    print("Running StudyPal Trustworthiness Benchmark...")
    report = run_trustworthiness_benchmark(
        project_root=PROJECT_ROOT,
        abstention_threshold=args.threshold,
        include_retrieval=not args.skip_retrieval,
    )

    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    paths = save_benchmark_report(report, reports_dir)

    sec = report["security_and_fairness"]
    priv = report["privacy"]
    faith = report["faithfulness_verifier"]
    ret = report.get("retrieval_and_abstention")

    print("\n================ TRUSTWORTHINESS BENCHMARK SUMMARY ================")
    print(f"Overall Benchmark Score        : {report['overall_trustworthiness_benchmark_score']:.1%}")
    print(f"Red-Team Attack Block Rate     : {sec['attack_block_rate']:.1%} ({sec['attacks_blocked']}/{sec['attacks_total']})")
    print(f"Benign Domain Allow Rate       : {1.0 - sec['false_positive_rate']:.1%} ({sec['benign_allowed']}/{sec['benign_total']})")
    print(f"PII Redaction Accuracy         : {priv['pii_accuracy']:.1%} ({priv['passed']}/{priv['total']})")
    print(
        f"Faithfulness Separation        : {faith['faithful_answer_score']:.2f} vs "
        f"{faith['hallucinated_answer_score']:.2f} (margin +{faith['separation_margin']:.2f}, "
        f"invalid citation caught={faith['caught_fabricated_citation']})"
    )
    if ret is not None:
        print(
            f"In-Domain Retrieval Hit Rate   : {ret['in_domain_hit_rate']:.1%} "
            f"(avg top sim {ret['in_domain_avg_top_score']:.2f})"
        )
        print(
            f"Out-of-Context Abstention Rate : {ret['out_of_context_abstention_rate']:.1%} "
            f"(avg top sim {ret['out_of_context_avg_top_score']:.2f})"
        )
    print("===================================================================")
    print(f"JSON Report     : {paths['json']}")
    print(f"Markdown Report : {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
