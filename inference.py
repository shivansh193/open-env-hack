"""
inference.py
------------
Baseline agent script for the Synthetic Market RL Environment.
Must be run from the project root directory.

Usage:
    API_BASE_URL=<url> MODEL_NAME=<model> API_KEY=<token> python inference.py

Environment variables (required):
    API_BASE_URL   — LLM API base URL
    MODEL_NAME     — model identifier
    API_KEY        — API key (injected by evaluator)
    HF_TOKEN       — fallback API key for local testing

Stdout log format (strictly followed for evaluation):
    [START] {"type": "START", "task_id": ..., "step": 0}
    [STEP]  {"type": "STEP",  "step": ..., "action": ..., "reasoning": ...}
    [END]   {"type": "END",   "task_id": ..., "total_reward": ..., "score": ...}
"""

import os
import sys
import json
import time
import requests
from typing import List, Optional
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN     = os.getenv("API_KEY", os.getenv("HF_TOKEN", ""))

ENV_URL   = os.environ.get("ENV_URL", "http://127.0.0.1:8000")
BENCHMARK = "synthetic-market-env"

# ---------------------------------------------------------------------------
# History window — keep system prompt + last N turns to stay within token limits
# ---------------------------------------------------------------------------
MAX_HISTORY_TURNS = 6   # 3 user/assistant exchanges = ~1500-2000 tokens

# ---------------------------------------------------------------------------
# System prompts (one per task)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "task_1": """You are a portfolio manager trading a single synthetic asset (asset_A).
You receive market state every step and must return a JSON allocation decision.

MARKET MECHANICS (hidden from you but real):
- The market alternates between TRENDING (positive momentum) and MEAN-REVERTING (negative momentum) regimes.
- Volatility clusters: high vol follows high vol. The volatility_signal (0-1) tells you how choppy it is.
- Your job: detect the regime from price behaviour and size positions accordingly.

YOUR ACTION must be valid JSON with exactly this structure:
{"asset_A": <float between -1.0 and 1.0>, "reasoning": "<brief reasoning>"}

Positive = long (profit if price rises). Negative = short (profit if price falls). 0 = flat.

RISK RULES:
- Drawdown beyond 10% incurs heavy penalties. Protect capital first.
- Transaction costs apply on every position change — don't churn.
- Scale down position size when volatility_signal is high (>0.7).

REGIME SIGNALS to watch:
- Trending: recent returns are consistently positive or consistently negative (momentum).
- Mean-reverting: recent returns alternate sign (bounces).
- Adapt your bias: momentum trade in trending, fade extremes in mean-reverting.

Respond ONLY with the JSON object. No preamble, no explanation outside the JSON.""",

    "task_2": """You are a portfolio manager trading two synthetic assets (asset_A, asset_B).
They are correlated — you must estimate the correlation from observed price behaviour.

YOUR ACTION must be valid JSON:
{"asset_A": <float -1 to 1>, "asset_B": <float -1 to 1>, "reasoning": "<brief reasoning>"}

Sum of |allocations| is capped at 1.5 by the environment (leverage limit).

NEWS: Headlines arrive occasionally. Learn the direction:
- Monetary tightening → both assets negative
- Geopolitical risk → A negative, B positive (flight to safety)
- Positive output data → both positive
- Sector investigation → that sector's asset negative

REGIME: The market switches regime at least once per episode. Watch for the flip:
- Momentum working → trending. Fade/contrarian working → mean-reverting.
- When you detect a switch, reverse your directional bias quickly.

RISK RULES:
- Drawdown >15% is penalised heavily. Keep aggregate exposure lower in volatile markets.
- Use the correlation: if assets move together, don't double up — hedge instead.

Respond ONLY with the JSON object.""",

    "task_3": """You are a portfolio manager trading four synthetic assets (asset_A, asset_B, asset_C, asset_D).
Assets come in two correlated pairs: (A,B) and (C,D). Cross-pair correlation is low.

YOUR ACTION must be valid JSON:
{"asset_A": <float -1 to 1>, "asset_B": <float -1 to 1>,
 "asset_C": <float -1 to 1>, "asset_D": <float -1 to 1>,
 "reasoning": "<brief reasoning>"}

Sum of |allocations| is capped at 1.5 by the environment.

THIS IS THE HARD TASK:
- Starts in a volatile, mean-reverting regime.
- News shocks fire frequently (~every 6 steps).
- If drawdown exceeds 20%, the episode TERMINATES IMMEDIATELY with score 0.0.

SURVIVAL IS THE PRIORITY. Do not be a hero.

STRATEGY:
1. Start with low total exposure (sum of |allocs| < 0.8) until you read the market.
2. Reduce exposure aggressively when drawdown > 12% — get flat if needed.
3. Use pair structure: hedge A with B (same pair), or C with D, to reduce idiosyncratic risk.
4. Respond to news direction immediately but size conservatively (magnitude is uncertain).
5. High volatility_signal (>0.8) → cut all positions by 50%.

Respond ONLY with the JSON object.""",
}

# ---------------------------------------------------------------------------
# Action parsing helpers
# ---------------------------------------------------------------------------

ASSET_KEYS_BY_TASK = {
    "task_1": ["asset_A"],
    "task_2": ["asset_A", "asset_B"],
    "task_3": ["asset_A", "asset_B", "asset_C", "asset_D"],
}

DEFAULT_ACTIONS = {
    "task_1": {"asset_A": 0.0, "reasoning": "parse error fallback"},
    "task_2": {"asset_A": 0.0, "asset_B": 0.0, "reasoning": "parse error fallback"},
    "task_3": {"asset_A": 0.0, "asset_B": 0.0, "asset_C": 0.0, "asset_D": 0.0,
               "reasoning": "parse error fallback"},
}


def parse_action(text: str, task_id: str) -> dict:
    """
    Extract JSON action from LLM response text.
    Falls back to flat (zero) position on any parse failure.
    """
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    try:
        action = json.loads(text)
        # Validate required keys present
        required = ASSET_KEYS_BY_TASK[task_id]
        for key in required:
            if key not in action:
                action[key] = 0.0
        return action
    except (json.JSONDecodeError, KeyError):
        return DEFAULT_ACTIONS[task_id].copy()


def build_user_message(state: dict, task_id: str) -> str:
    """
    Format state into a compact, information-dense prompt for the LLM.
    Highlights the most actionable signals.
    """
    step     = state["step"]
    prices   = state["prices"]
    vol      = state["volatility_signal"]
    news     = state.get("news")
    position = state["position"]
    pnl      = state["episode_pnl"]
    dd       = state["drawdown"]
    cash     = state["cash"]
    step_ret = state.get("step_return", 0.0)

    # Compute recent return momentum per asset
    momentum_lines = []
    for asset_key, price_history in prices.items():
        if len(price_history) >= 3:
            recent = price_history[-3:]
            rets   = [(recent[i] - recent[i-1]) / recent[i-1] for i in range(1, len(recent))]
            trend  = "↑↑" if all(r > 0 for r in rets) else \
                     "↓↓" if all(r < 0 for r in rets) else \
                     "↕"  # alternating = mean-reverting signal
            momentum_lines.append(
                f"  {asset_key}: price={price_history[-1]:.4f}  "
                f"2-step returns=[{rets[0]:+.4f}, {rets[1]:+.4f}]  pattern={trend}"
            )

    momentum_str = "\n".join(momentum_lines)
    news_str     = f"\nNEWS THIS STEP: {news}" if news else ""
    position_str = "  " + ", ".join(
        f"{k}={v:+.4f} shares" for k, v in position.items()
    )

    # Drawdown warning
    dd_warn = ""
    if dd > 0.15:
        dd_warn = f"\n⚠️  DRAWDOWN WARNING: {dd:.1%} — reduce exposure NOW"
    elif dd > 0.10:
        dd_warn = f"\n⚠️  Drawdown at {dd:.1%} — be cautious"

    msg = f"""Step {step}/60  |  Vol signal: {vol:.3f}  |  Episode PnL: {pnl:+.2f}  |  Drawdown: {dd:.3%}{dd_warn}
{news_str}
Asset prices & momentum:
{momentum_str}

Current positions:
{position_str}
Cash: {cash:.2f}  |  Last step return: {step_ret:+.6f}

Decide your target allocation for the next step."""

    return msg.strip()


def trim_history(messages: list, max_turns: int) -> list:
    """
    Keep the system prompt (messages[0]) and the last max_turns messages.
    This prevents the context window from growing unboundedly across 60 steps.
    Each turn = 1 user message + 1 assistant message, so max_turns=6 keeps 3 exchanges.
    """
    if len(messages) <= max_turns + 1:   # +1 for system prompt
        return messages
    return [messages[0]] + messages[-(max_turns):]


# ---------------------------------------------------------------------------
# Logging helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> float:
    """
    Run one complete episode for the given task.
    Emits [START], [STEP], [END] logs to stdout.
    Returns the grader score.
    """
    # Initialise client here so it picks up env vars injected at runtime
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"]
    )

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    # Reset environment
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "seed": 42},
        timeout=30,
    )
    resp.raise_for_status()
    data  = resp.json()
    state = data["state"]

    messages = [{"role": "system", "content": SYSTEM_PROMPTS[task_id]}]
    rewards  = []
    score    = 0.0

    while True:
        # Build user message
        user_msg = build_user_message(state, task_id)
        messages.append({"role": "user", "content": user_msg})

        # Trim history to stay within token limits
        messages = trim_history(messages, MAX_HISTORY_TURNS)

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=200,
                temperature=0.2,
            )
            action_text = response.choices[0].message.content
        except Exception as e:
            # On LLM failure, go flat
            action_text = json.dumps(DEFAULT_ACTIONS[task_id])
            print(f"[WARN] LLM call failed at step {state['step']}: {e}", file=sys.stderr)

        # Parse action
        action = parse_action(action_text, task_id)

        # Compact action string for logs
        action_str = ",".join(f"{k}:{v:+.2f}" for k, v in action.items() if k != "reasoning")

        # Step environment
        step_resp = requests.post(
            f"{ENV_URL}/step",
            json={"action": action},
            timeout=30,
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        state  = result["state"]
        reward = result["reward"]
        done   = result["done"]
        info   = result["info"]
        rewards.append(reward)

        # [STEP] log (mandatory: immediately after env.step())
        log_step(step=state["step"], action=action_str, reward=reward, done=done, error=None)

        # Append assistant turn to conversation history
        messages.append({"role": "assistant", "content": action_text})

        if done:
            score = info.get("grader_score", 0.0)
            break

    # [END] log
    success = score >= 0.5
    log_end(success=success, steps=state["step"], score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tasks  = ["task_1", "task_2", "task_3"]
    scores = {}

    for task_id in tasks:
        print(f"\n{'='*60}", flush=True)
        print(f"Running {task_id}...", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            score = run_episode(task_id)
            scores[task_id] = score
        except Exception as e:
            print(f"[END] success=false steps=0 score=0.000 rewards=", flush=True)
            print(f"[ERROR] {task_id} failed: {e}", file=sys.stderr)
            scores[task_id] = 0.0

        # Brief pause between tasks
        time.sleep(1)

    print("\n" + "="*60, flush=True)
    print("FINAL SCORES", flush=True)
    print("="*60, flush=True)
    for tid, s in scores.items():
        print(f"  {tid}: {s:.4f}", flush=True)
    print(f"  average: {sum(scores.values()) / len(scores):.4f}", flush=True)


if __name__ == "__main__":
    main()