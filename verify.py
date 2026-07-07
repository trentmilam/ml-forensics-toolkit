"""Root-level aggregate verifier: runs both rag-toolkit experiments (offline,
deterministic, no network) plus the pytest suite, and prints one PASS/FAIL for
the whole repo.

Usage:
    python verify.py
"""
from __future__ import annotations
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def _run(cmd, cwd, env):
    print(f"\n$ {' '.join(cmd)}  (cwd={cwd})")
    return subprocess.run(cmd, cwd=cwd, env=env).returncode == 0


def main():
    env = dict(os.environ)
    env["RAGTOOLKIT_OFFLINE"] = "1"  # deterministic, network-free run of both tools

    ok_qslm = _run([sys.executable, "run_experiment.py"], os.path.join(ROOT, "qslm"), env)
    ok_bdapp = _run([sys.executable, "run_experiment.py"], os.path.join(ROOT, "bdapp"), env)
    ok_tests = _run([sys.executable, "-m", "pytest", "-q"], ROOT, env)

    print("\n" + "=" * 60)
    print(f"qslm/run_experiment.py  : {'PASS' if ok_qslm else 'FAIL'}")
    print(f"bdapp/run_experiment.py : {'PASS' if ok_bdapp else 'FAIL'}")
    print(f"pytest (tests/)         : {'PASS' if ok_tests else 'FAIL'}")
    all_ok = ok_qslm and ok_bdapp and ok_tests
    print(f"RESULT: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
