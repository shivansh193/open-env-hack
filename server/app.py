"""
server.py
---------
FastAPI HTTP server exposing the OpenEnv-compliant API:

    POST /reset   — start new episode
    POST /step    — take action, advance environment
    GET  /state   — current state without advancing
    GET  /ping    — health check (returns 200)
    GET  /tasks   — list available tasks
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from server.environment import MarketEnvironment, TASK_CONFIGS


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = Field("task_1", description="One of task_1, task_2, task_3")
    seed: Optional[int] = Field(None, description="RNG seed for reproducibility")


class StepRequest(BaseModel):
    action: Dict[str, Any] = Field(
        ...,
        description=(
            "Allocation dict. Keys: asset_A [, asset_B, asset_C, asset_D], reasoning. "
            "Values: float in [-1, 1]. reasoning is optional str."
        ),
    )


class PingResponse(BaseModel):
    status: str = "ok"


class TaskInfo(BaseModel):
    id:          str
    description: str
    n_assets:    int
    max_steps:   int
    difficulty:  str


class TaskListResponse(BaseModel):
    tasks: list[TaskInfo]


# ---------------------------------------------------------------------------
# App + shared environment instance
# ---------------------------------------------------------------------------

env: MarketEnvironment = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    env = MarketEnvironment()
    yield


app = FastAPI(
    title="Synthetic Market RL Environment",
    version="1.0.0",
    description=(
        "OpenEnv-compliant synthetic financial market environment. "
        "LLM agents act as portfolio managers making sequential trading decisions."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_reset():
    if env is None or env._market_snap is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. Call POST /reset first.",
        )


def _strip_internal(state: dict) -> dict:
    """Remove internal-only fields before sending to agent."""
    return {k: v for k, v in state.items()
            if k not in ("current_regime", "raw_variances")}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/ping", response_model=PingResponse, tags=["health"])
def ping():
    """Health check — judges ping this to verify the Space is live."""
    return PingResponse(status="ok")


@app.get("/tasks", response_model=TaskListResponse, tags=["meta"])
def list_tasks():
    """Enumerate all available tasks with metadata."""
    difficulty_map = {"task_1": "easy", "task_2": "medium", "task_3": "hard"}
    tasks = [
        TaskInfo(
            id=tid,
            description=cfg["description"],
            n_assets=cfg["n_assets"],
            max_steps=cfg["max_steps"],
            difficulty=difficulty_map[tid],
        )
        for tid, cfg in TASK_CONFIGS.items()
    ]
    return TaskListResponse(tasks=tasks)


@app.post("/reset", tags=["env"])
def reset(body: ResetRequest = ResetRequest()):
    """
    Start a new episode.

    Returns the initial state dict. No positions are open at step 0.
    Pass seed=42 for the reproducible baseline run.
    Body is optional — defaults to task_1 with no seed.
    """
    global env
    try:
        state = env.reset(task_id=body.task_id, seed=body.seed)
    except AssertionError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "state":   _strip_internal(state),
        "task_id": body.task_id,
        "seed":    body.seed,
    }


@app.post("/step", tags=["env"])
def step(body: StepRequest):
    """
    Apply an action and advance the environment by one step.

    Action keys: asset_A (required), asset_B / asset_C / asset_D (task-dependent),
    reasoning (optional str, logged but not used in grading).

    Returns state, reward, done, info.
    On the final step (done=True), info contains grader_score and sharpe.
    """
    _require_reset()

    if env.episode_done:
        raise HTTPException(
            status_code=400,
            detail="Episode finished. Call POST /reset to start a new one.",
        )

    try:
        state, reward, done, info = env.step(body.action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step error: {e}")

    return {
        "state":  _strip_internal(state),
        "reward": reward,
        "done":   done,
        "info":   info,
    }


@app.get("/state", tags=["env"])
def get_state():
    """
    Return the current environment state without advancing it.
    Useful for agents that want to re-read state before acting.
    """
    _require_reset()

    try:
        state = env.get_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"state": _strip_internal(state)}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()