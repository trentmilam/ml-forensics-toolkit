"""Root-level aggregate verifier: runs both ml-forensics-toolkit experiments (offline,
deterministic, no network) plus the pytest suite, and prints one PASS/FAIL for
the whole repo.

Usage:
    python verify.py
"""
from __future__ import annotations
import importlib.util
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

    # pytest is an optional *dev* dependency (requirements-dev.txt), not required to run
    # the two experiments above. If it's not installed, skip the test suite explicitly
    # instead of letting a bare "No module named pytest" traceback read as a toolkit
    # failure in the final banner.
    if importlib.util.find_spec("pytest") is not None:
        ok_tests = _run([sys.executable, "-m", "pytest", "-q"], ROOT, env)
        tests_line = "PASS" if ok_tests else "FAIL"
    else:
        ok_tests = True
        tests_line = "SKIPPED (pytest not installed -- optional dev dependency; run " \
                     "`pip install -r requirements-dev.txt` to enable)"
        print("\npytest not installed -- skipping the test suite (optional dev dependency,")
        print("not required for the toolkit itself). Install with:")
        print("    pip install -r requirements-dev.txt")

    print("\n" + "=" * 60)
    print(f"qslm/run_experiment.py  : {'PASS' if ok_qslm else 'FAIL'}")
    print(f"bdapp/run_experiment.py : {'PASS' if ok_bdapp else 'FAIL'}")
    print(f"pytest (tests/)         : {tests_line}")
    all_ok = ok_qslm and ok_bdapp and ok_tests
    print(f"RESULT: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
