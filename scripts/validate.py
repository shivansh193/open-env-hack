"""
validate.py
-----------
Pre-submission validation script. Run this before submitting to catch
any issues that would cause disqualification.

Usage:
    # With server already running:
    ENV_URL=http://127.0.0.1:8000 python scripts/validate.py

    # Or let this script start the server itself:
    python scripts/validate.py --start-server

Exit code 0 = all checks passed. Non-zero = failures found.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import requests

ENV_URL = os.environ.get("ENV_URL", "http://127.0.0.1:8000")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

failures = []
warnings = []


def ok(msg):
    print(f"  {PASS} {msg}")


def fail(msg):
    print(f"  {FAIL} {msg}")
    failures.append(msg)


def warn(msg):
    print(f"  {WARN} {msg}")
    warnings.append(msg)


def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_files():
    section("Required files")
    required = [
        ("inference.py",    "inference script in root"),
        ("openenv.yaml",    "environment spec"),
        ("Dockerfile",      "container definition"),
        ("README.md",       "documentation"),
        ("requirements.txt","dependencies"),
        ("server/app.py",   "FastAPI server"),
        ("server/environment.py", "server.environment.MarketEnvironment._grade logic"),
        ("server/market_engine.py", "market engine"),
    ]
    for path, label in required:
        if os.path.exists(path):
            ok(f"{label} ({path})")
        else:
            fail(f"Missing: {path} — {label}")


def check_inference_script():
    section("inference.py structure")

    if not os.path.exists("inference.py"):
        fail("inference.py not found — skipping structure checks")
        return

    with open("inference.py", encoding="utf-8") as f:
        src = f.read()

    checks = [
        ("import requests",           "requests imported"),
        ("os.environ[\"API_BASE_URL\"]", "API_BASE_URL read from env"),
        ("os.environ[\"MODEL_NAME\"]",   "MODEL_NAME read from env"),
        ("os.environ[\"HF_TOKEN\"]",     "HF_TOKEN read from env"),
        ("OpenAI(",                    "OpenAI client instantiated"),
        ("log_start(",                 "log_start function used"),
        ("log_step(",                  "log_step function used"),
        ("log_end(",                   "log_end function used"),
        ('f"[START] task=',            "[START] format correct"),
        ('f"[STEP] step=',             "[STEP] format correct"),
        ('f"[END] success=',           "[END] format correct"),
        ("BENCHMARK =",                "BENCHMARK constant defined"),
        ("task_1",                     "task_1 referenced"),
        ("task_2",                     "task_2 referenced"),
        ("task_3",                     "task_3 referenced"),
    ]
    for snippet, label in checks:
        if snippet in src:
            ok(label)
        else:
            fail(f"Not found in inference.py: {label!r}")


def check_openenv_yaml():
    section("openenv.yaml")

    if not os.path.exists("openenv.yaml"):
        fail("openenv.yaml not found — skipping")
        return

    try:
        import yaml
    except ImportError:
        warn("PyYAML not installed — skipping yaml parse check")
        return

    with open("openenv.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_top = ["name", "version", "description", "tasks",
                    "observation_space", "action_space"]
    for key in required_top:
        if key in cfg:
            ok(f"Top-level key: {key}")
        else:
            fail(f"Missing top-level key in openenv.yaml: {key}")

    tasks = cfg.get("tasks", [])
    if len(tasks) >= 3:
        ok(f"{len(tasks)} tasks defined")
    else:
        fail(f"Need at least 3 tasks, found {len(tasks)}")

    difficulties = {t.get("difficulty") for t in tasks}
    for d in ["easy", "medium", "hard"]:
        if d in difficulties:
            ok(f"Difficulty level present: {d}")
        else:
            fail(f"Missing difficulty level: {d}")


def check_server_ping():
    section("Server health")
    try:
        r = requests.get(f"{ENV_URL}/ping", timeout=10)
        if r.status_code == 200:
            ok(f"GET /ping → 200")
        else:
            fail(f"GET /ping returned {r.status_code}")
    except requests.exceptions.ConnectionError:
        fail(f"Cannot reach server at {ENV_URL} — is it running?")
        return False
    return True


def check_tasks_endpoint():
    section("GET /tasks")
    try:
        r = requests.get(f"{ENV_URL}/tasks", timeout=10)
        if r.status_code != 200:
            fail(f"/tasks returned {r.status_code}")
            return
        data  = r.json()
        tasks = data.get("tasks", [])
        if len(tasks) >= 3:
            ok(f"{len(tasks)} tasks returned")
        else:
            fail(f"Expected 3+ tasks, got {len(tasks)}")
        for t in tasks:
            for key in ["id", "description", "n_assets", "max_steps", "difficulty"]:
                if key not in t:
                    fail(f"Task missing field: {key}")
            ok(f"Task {t['id']}: difficulty={t['difficulty']}, assets={t['n_assets']}")
    except Exception as e:
        fail(f"/tasks error: {e}")


def check_reset():
    section("POST /reset")
    results = {}
    for task_id in ["task_1", "task_2", "task_3"]:
        try:
            r = requests.post(
                f"{ENV_URL}/reset",
                json={"task_id": task_id, "seed": 42},
                timeout=10,
            )
            if r.status_code != 200:
                fail(f"/reset {task_id} → {r.status_code}")
                continue
            data  = r.json()
            state = data.get("state", {})

            required_fields = [
                "step", "prices", "volume", "volatility_signal",
                "news", "position", "cash", "unrealized_pnl",
                "episode_pnl", "drawdown", "step_return",
            ]
            missing = [f for f in required_fields if f not in state]
            if missing:
                fail(f"{task_id} state missing fields: {missing}")
            else:
                ok(f"{task_id}: all state fields present")

            # Internal fields must NOT be exposed
            internal = ["current_regime", "raw_variances"]
            leaked   = [f for f in internal if f in state]
            if leaked:
                fail(f"{task_id} leaks internal fields: {leaked}")
            else:
                ok(f"{task_id}: no internal fields leaked")

            results[task_id] = state
        except Exception as e:
            fail(f"/reset {task_id} error: {e}")
    return results


def check_step(reset_results):
    section("POST /step")
    actions = {
        "task_1": {"asset_A": 0.3, "reasoning": "validation test"},
        "task_2": {"asset_A": 0.3, "asset_B": 0.2, "reasoning": "validation test"},
        "task_3": {"asset_A": 0.2, "asset_B": -0.1,
                   "asset_C": 0.15, "asset_D": -0.1, "reasoning": "validation test"},
    }
    for task_id, action in actions.items():
        try:
            # Reset first
            requests.post(f"{ENV_URL}/reset",
                          json={"task_id": task_id, "seed": 42}, timeout=10)
            r = requests.post(f"{ENV_URL}/step",
                              json={"action": action}, timeout=10)
            if r.status_code != 200:
                fail(f"/step {task_id} → {r.status_code}: {r.text[:100]}")
                continue
            data = r.json()
            for key in ["state", "reward", "done", "info"]:
                if key not in data:
                    fail(f"{task_id} /step response missing: {key}")
            if data["state"]["step"] == 1:
                ok(f"{task_id}: step advances correctly")
            else:
                fail(f"{task_id}: expected step=1, got {data['state']['step']}")
            if isinstance(data["reward"], (int, float)):
                ok(f"{task_id}: reward is numeric ({data['reward']:.6f})")
            else:
                fail(f"{task_id}: reward is not numeric")
        except Exception as e:
            fail(f"/step {task_id} error: {e}")


def check_get_state():
    section("GET /state")
    try:
        requests.post(f"{ENV_URL}/reset",
                      json={"task_id": "task_1", "seed": 42}, timeout=10)
        r1 = requests.get(f"{ENV_URL}/state", timeout=10)
        r2 = requests.get(f"{ENV_URL}/state", timeout=10)
        if r1.status_code == 200 and r2.status_code == 200:
            s1 = r1.json()["state"]["step"]
            s2 = r2.json()["state"]["step"]
            if s1 == s2 == 0:
                ok("GET /state does not advance episode")
            else:
                fail(f"GET /state advanced episode: step went {s1} → {s2}")
        else:
            fail(f"GET /state returned non-200: {r1.status_code}")
    except Exception as e:
        fail(f"GET /state error: {e}")


def check_full_episode_grader():
    section("Full episode + grader scores")
    actions = {
        "task_1": {"asset_A": 0.4},
        "task_2": {"asset_A": 0.3, "asset_B": 0.2},
        "task_3": {"asset_A": 0.2, "asset_B": 0.1, "asset_C": 0.15, "asset_D": 0.05},
    }
    for task_id, action in actions.items():
        try:
            requests.post(f"{ENV_URL}/reset",
                          json={"task_id": task_id, "seed": 42}, timeout=10)
            result = None
            for _ in range(60):
                r = requests.post(f"{ENV_URL}/step",
                                  json={"action": action}, timeout=10)
                result = r.json()
                if result.get("done"):
                    break

            info  = result.get("info", {})
            score = info.get("grader_score")
            if score is None:
                fail(f"{task_id}: grader_score missing from final info")
            elif 0.0 <= score <= 1.0:
                ok(f"{task_id}: grader_score={score:.4f} (valid range [0, 1])")
            else:
                fail(f"{task_id}: grader_score={score} out of [0, 1]")
        except Exception as e:
            fail(f"{task_id} full episode error: {e}")


def check_reproducibility():
    section("Reproducibility (seed=42)")
    action = {"asset_A": 0.4}

    def run():
        requests.post(f"{ENV_URL}/reset",
                      json={"task_id": "task_1", "seed": 42}, timeout=10)
        result = None
        for _ in range(60):
            r = requests.post(f"{ENV_URL}/step",
                              json={"action": action}, timeout=10)
            result = r.json()
            if result.get("done"):
                break
        return result["info"].get("grader_score"), result["info"].get("sharpe")

    try:
        s1, sh1 = run()
        s2, sh2 = run()
        if s1 == s2 and sh1 == sh2:
            ok(f"Scores identical across two runs: score={s1}, sharpe={sh1}")
        else:
            fail(f"Non-reproducible: run1=({s1},{sh1}) run2=({s2},{sh2})")
    except Exception as e:
        fail(f"Reproducibility check error: {e}")


def check_leverage_clipping():
    section("Leverage clipping (sum |allocs| > 1.5)")
    try:
        requests.post(f"{ENV_URL}/reset",
                      json={"task_id": "task_3", "seed": 1}, timeout=10)
        # Send over-leveraged action — should not crash
        big = {"asset_A": 1.0, "asset_B": 1.0, "asset_C": 1.0, "asset_D": 1.0}
        r   = requests.post(f"{ENV_URL}/step", json={"action": big}, timeout=10)
        if r.status_code == 200:
            ok("Over-leveraged action accepted and clipped (no 500 error)")
        else:
            fail(f"Over-leveraged action caused {r.status_code}: {r.text[:100]}")
    except Exception as e:
        fail(f"Leverage clipping check error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global ENV_URL
    parser = argparse.ArgumentParser(description="OpenEnv pre-submission validator")
    parser.add_argument("--start-server", action="store_true",
                        help="Start uvicorn server before running checks")
    parser.add_argument("--env-url", default=ENV_URL,
                        help=f"Environment server URL (default: {ENV_URL})")
    args = parser.parse_args()

    ENV_URL = args.env_url

    print("\n" + "="*55)
    print("  OpenEnv Pre-Submission Validator")
    print("="*55)
    print(f"  Server URL: {ENV_URL}")

    server_proc = None
    if args.start_server:
        print("\n  Starting server...")
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app",
             "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(4)

    # File checks (no server needed)
    check_files()
    check_inference_script()
    check_openenv_yaml()

    # Server checks
    server_ok = check_server_ping()
    if server_ok:
        check_tasks_endpoint()
        reset_results = check_reset()
        check_step(reset_results)
        check_get_state()
        check_full_episode_grader()
        check_reproducibility()
        check_leverage_clipping()
    else:
        warn("Skipping server-dependent checks — server not reachable")

    # Summary
    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    if failures:
        print(f"\n  {FAIL} {len(failures)} check(s) failed:\n")
        for f in failures:
            print(f"    • {f}")
    if warnings:
        print(f"\n  {WARN} {len(warnings)} warning(s):\n")
        for w in warnings:
            print(f"    • {w}")
    if not failures:
        print(f"\n  {PASS} All checks passed! Ready to submit.\n")
    else:
        print(f"\n  Fix the above issues before submitting.\n")

    if server_proc:
        server_proc.terminate()

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()