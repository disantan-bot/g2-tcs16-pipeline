"""
R16.7 verifier.

Run:
    python run_verifier.py

Outputs:
    outputs/results_R16_7.json
    outputs/report_R16_7.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from toe_mre.core import (
    CAConfig,
    exhaustive_permutation_check,
    reversibility_trials,
    trajectory_observables,
)
from toe_mre.geometry import spectral_dimension_summary


def classify(results: dict) -> dict:
    """Classify claims as Green, Amber, or Open based on the verifier results."""
    green = []
    amber = []
    open_claims = []

    if results["permutation_check"]["is_permutation"]:
        green.append("Finite-state update T is a permutation for the fully enumerated N=2 state space.")
    else:
        open_claims.append("Finite-state update failed the exhaustive permutation check.")

    if results["reversibility_trials"]["passed"]:
        green.append("Forward evolution followed by exact inverse recovers initial states for all random trials.")
    else:
        open_claims.append("At least one random reversibility trial failed.")

    if results["geometry"]["within_broad_tolerance"]:
        amber.append(
            "Spectral geometry pipeline recovers a dimension near 2 for a planted 2D torus."
        )
    else:
        open_claims.append("Spectral dimension sanity check did not return a value near 2.")

    amber.append(
        "Information conservation is verified here as finite reversibility/bijection, not as a full physical derivation."
    )

    open_claims.extend([
        "Derivation of Standard Model field content is not implemented.",
        "Derivation of physical constants is not implemented.",
        "Emergent Lorentzian spacetime/gravity is not implemented.",
        "A new falsifiable physical prediction is not implemented.",
    ])

    return {"green": green, "amber": amber, "open": open_claims}


def markdown_report(results: dict, ledger: dict) -> str:
    """Create a compact markdown report."""
    lines = []
    lines.append("# TOE R16.7 — Verifier Report")
    lines.append("")
    lines.append(f"Generated UTC: {results['generated_utc']}")
    lines.append("")
    lines.append("## Core results")
    lines.append("")
    lines.append(f"- Exhaustive permutation check N={results['permutation_check']['n']}: "
                 f"{results['permutation_check']['is_permutation']} "
                 f"({results['permutation_check']['image_count']}/"
                 f"{results['permutation_check']['state_count']} unique images)")
    lines.append(f"- Entropy before/after under uniform finite ensemble: "
                 f"{results['permutation_check']['uniform_entropy_bits_before']:.6f} / "
                 f"{results['permutation_check']['uniform_entropy_bits_after']:.6f} bits")
    lines.append(f"- Random reversibility trials: {results['reversibility_trials']['passed']} "
                 f"({results['reversibility_trials']['seeds']} seeds, "
                 f"{results['reversibility_trials']['steps']} steps)")
    lines.append(f"- Spectral dimension median central estimate: "
                 f"{results['geometry']['spectral_dimension_median_central']:.6f}")
    lines.append("")
    lines.append("## Claim ledger")
    lines.append("")
    lines.append("### Green")
    for item in ledger["green"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("### Amber")
    for item in ledger["amber"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("### Open")
    for item in ledger["open"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "R16.7 establishes a reproducible computational kernel for the information-conservation "
        "postulate in the narrow finite-state sense. It does not solve the TOE. The geometry module "
        "is a sanity check for the extraction pipeline, because the tested geometry is planted rather "
        "than derived."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    results = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "permutation_check": exhaustive_permutation_check(n=2),
        "reversibility_trials": reversibility_trials(CAConfig(n=16, steps=64, seeds=25)),
        "trajectory_observables": trajectory_observables(n=16, steps=64, seed=0),
        "geometry": spectral_dimension_summary(n=32),
    }

    ledger = classify(results)
    results["claim_ledger"] = ledger

    (out / "results_R16_7.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "report_R16_7.md").write_text(
        markdown_report(results, ledger),
        encoding="utf-8",
    )

    print(markdown_report(results, ledger))


if __name__ == "__main__":
    main()