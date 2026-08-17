"""Audit solver implementations and numerical methods for multi-solver analysis.

This script inventories:
1. Solver-level classes in the vendored Genesis copy and in the active installed
   genesis-world package.
2. Numerical-method keywords found in each solver file.
3. A simple matrix that can be used as a starting point for energy/momentum
   conservation analysis across coupling boundaries.

Run with the project venv:
    uv run python tools/audit_solvers.py
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories that contain physics/constraint/collision solver implementations.
SOLVER_ROOTS = {
    "vendored": REPO_ROOT
    / "plugins"
    / "envs"
    / "sky"
    / "core"
    / "genesis"
    / "engine"
    / "solvers",
    "active": REPO_ROOT
    / ".venv"
    / "Lib"
    / "site-packages"
    / "genesis"
    / "engine"
    / "solvers",
}

OUTPUT_DIR = REPO_ROOT / "outputs" / "solver_audit"

# Numerical-method keywords. Each regex is matched case-insensitively.
METHOD_KEYWORDS: dict[str, re.Pattern] = {
    # Time integration
    "explicit_euler": re.compile(r"\bexplicit[_-]?euler\b|\bforward[_-]?euler\b"),
    "implicit_euler": re.compile(r"\bimplicit[_-]?euler\b"),
    "implicitfast": re.compile(r"\bimplicitfast\b|\bapproximate_implicitfast\b"),
    "symplectic_euler": re.compile(r"\bsymplectic\b|\beuler[_-]?cromer\b"),
    "runge_kutta": re.compile(r"\brunge[_-]?kutta\b|\brk4\b"),
    "ftcs": re.compile(r"\bftcs\b"),
    "fdtd_yee": re.compile(r"\bfdtd\b|\byee\b"),
    "enthalpy_method": re.compile(r"\benthalpy\b"),
    # Hybrid Lagrangian-Eulerian / particle methods
    "pic_flip": re.compile(r"\bpic\b|\bflip\b|\bcpic\b"),
    "sph": re.compile(r"\bwcsph\b|\bdfsph\b|\bsph\b"),
    "position_based_dynamics": re.compile(r"\bxpbd\b|\bposition[_-]?based\b"),
    "finite_element": re.compile(r"\bfem\b|\bfinite[_-]?element\b"),
    "stable_fluids": re.compile(
        r"\bsemi[_-]?lagrangian\b|\bpressure[_-]?projection\b|\bstable[_-]?fluids\b"
    ),
    # Linear / nonlinear algebraic solvers
    "newton": re.compile(r"\bnewton\b|\bnewton[_-]?raphson\b"),
    "conjugate_gradient": re.compile(r"\bpcg\b|\bcg\b|\bconjugate[_-]?gradient\b"),
    "jacobi": re.compile(r"\bjacobi\b"),
    "gauss_seidel": re.compile(r"\bgauss[_-]?seidel\b"),
    "cholesky": re.compile(r"\bcholesky\b"),
    "svd": re.compile(r"\bsvd\b"),
    "line_search": re.compile(r"\bline[_-]?search\b|\bbacktracking\b"),
    "fixed_point": re.compile(r"\bfixed[_-]?point\b|\bfixed_point\b"),
    # Collision / geometry
    "mpr": re.compile(r"\bmpr\b|\bminkowski[_-]?portal\b"),
    "gjk": re.compile(r"\bgjk\b"),
    "epa": re.compile(r"\bepa\b"),
    "ccd": re.compile(r"\bccd\b|\bcontinuous[_-]?collision\b"),
    # Couplers / contact
    "sap": re.compile(r"\bsap\b|\bsemi[_-]?analytic[_-]?primal\b"),
    "ipc": re.compile(r"\bipc\b|\bincremental[_-]?potential\b"),
    "lcp": re.compile(r"\blcp\b|\blinear[_-]?complementarity\b"),
    # Control / optimization (project-level wrappers)
    "mpc": re.compile(r"\bmpc\b|\bmodel[_-]?predictive\b"),
    "wbc": re.compile(r"\bwbc\b|\bnull[_-]?space\b|\bpseudoinverse\b"),
    "ik": re.compile(
        r"\binverse[_-]?kinematics\b|\blevenberg[_-]?marquardt\b|\bdamped[_-]?least[_-]?squares\b"
    ),
    "ompl_rrt": re.compile(r"\bompl\b|\brrt\b|\brrtconnect\b"),
    "adam": re.compile(r"\badam\b"),
}


def find_python_files(root: Path) -> Iterable[Path]:
    """Yield non-empty Python files under ``root`` if it exists."""
    if not root.exists():
        return
    yield from sorted(p for p in root.rglob("*.py") if p.stat().st_size > 0)


def class_definitions(path: Path) -> list[tuple[str, int, list[str]]]:
    """Return (class_name, line_no, base_names) for each class in a file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases: list[str] = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    bases.append(b.attr)
                elif isinstance(b, ast.Subscript):
                    # e.g. Solver[SomeType]
                    if isinstance(b.value, ast.Name):
                        bases.append(b.value.id)
            out.append((node.name, node.lineno, bases))
    return out


def detect_methods(text: str) -> dict[str, bool]:
    """Return a mapping of numerical-method keywords to presence flags."""
    lowered = text.lower()
    return {name: bool(pat.search(lowered)) for name, pat in METHOD_KEYWORDS.items()}


def is_solverish(name: str, bases: list[str]) -> bool:
    """Heuristic: class is solver-level if its name contains Solver or base is Solver."""
    if "Solver" in name:
        return True
    if any("Solver" in b for b in bases):
        return True
    return name in {
        "MPR",
        "GJK",
        "Collider",
        "ConstraintSolver",
        "ConstraintSolverIsland",
    }


def audit_root(label: str, root: Path) -> dict:
    """Audit one solver root and return a summary dictionary."""
    files = list(find_python_files(root))
    solver_classes: list[dict] = []
    method_counter: Counter = Counter()
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(root).as_posix()
        methods = detect_methods(text)
        for cls_name, line_no, bases in class_definitions(path):
            if not is_solverish(cls_name, bases):
                continue
            cls_methods = {k: v for k, v in methods.items() if v}
            method_counter.update(cls_methods.keys())
            solver_classes.append(
                {
                    "file": rel,
                    "class": cls_name,
                    "line": line_no,
                    "bases": bases,
                    "methods": sorted(cls_methods.keys()),
                }
            )
    return {
        "label": label,
        "root": root.as_posix(),
        "exists": root.exists(),
        "files_scanned": len(files),
        "solver_classes": solver_classes,
        "unique_solver_classes": len({c["class"] for c in solver_classes}),
        "method_frequencies": dict(method_counter.most_common()),
    }


def write_csv_matrix(results: list[dict]) -> Path:
    """Write a solver-vs-method matrix CSV and return its path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "solver_method_matrix.csv"
    all_methods = sorted(
        {m for r in results for c in r["solver_classes"] for m in c["methods"]}
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "file", "class", "line", "bases"] + all_methods)
        for r in results:
            for c in r["solver_classes"]:
                row = [
                    r["label"],
                    c["file"],
                    c["class"],
                    c["line"],
                    ",".join(c["bases"]),
                ]
                row += ["x" if m in c["methods"] else "" for m in all_methods]
                writer.writerow(row)
    return csv_path


def main() -> int:
    """Run the solver audit across all configured roots."""
    results = []
    for label, root in SOLVER_ROOTS.items():
        result = audit_root(label, root)
        results.append(result)
        print(f"\n=== {label}: {root} (exists={result['exists']}) ===")
        if not result["exists"]:
            continue
        print(f"Files scanned: {result['files_scanned']}")
        print(
            f"Solver-level classes: {len(result['solver_classes'])} "
            f"(unique names: {result['unique_solver_classes']})"
        )
        print("\nClasses:")
        for c in result["solver_classes"]:
            bases = ",".join(c["bases"]) or "-"
            methods = ",".join(c["methods"]) or "-"
            print(
                f"  {c['class']:30s}  {c['file']:50s}  line {c['line']:4d}  bases={bases:20s}  methods={methods}"
            )
        print("\nNumerical-method keyword frequencies:")
        for method, count in result["method_frequencies"].items():
            print(f"  {method:25s}: {count}")

    csv_path = write_csv_matrix(results)
    json_path = OUTPUT_DIR / "solver_audit.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Summary ===")
    total_classes = sum(len(r["solver_classes"]) for r in results if r["exists"])
    total_unique = sum(r["unique_solver_classes"] for r in results if r["exists"])
    print(f"Total solver-level classes across scopes: {total_classes}")
    print(f"Total unique class names across scopes: {total_unique}")
    all_methods = sorted(
        {m for r in results for c in r["solver_classes"] for m in c["methods"]}
    )
    print(f"Distinct numerical-method keywords detected: {len(all_methods)}")
    print("\nArtifacts written:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
