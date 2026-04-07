---
title: Openenv Hack
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---
# Synthetic Market RL Environment

A synthetic financial market environment for LLM agents, built on the OpenEnv framework. The agent acts as a portfolio manager making sequential trading decisions over 60-step episodes.

**Core design principle:** simulate market *dynamics*, not market *data*. Price sequences are generated fresh every episode using real statistical models (GARCH volatility clustering, Hidden Markov regime switching), so the agent cannot exploit memorised historical patterns. Financial reasoning about volatility, momentum, and news impact is genuinely applicable — but lookahead bias is impossible.

---

## Observation Space

Each step the agent receives a state dictionary with the following fields:

| Field | Type | Description |
|---|---|---|
| `step` | `int` | Current step (0–59) |
| `prices` | `dict[str, list[float]]` | Last 10 closing prices per asset |
| `volume` | `dict[str, list[int]]` | Last 10 volume readings per asset |
| `volatility_signal` | `float [0, 1]` | Normalised volatility indicator. High = choppy market |
| `news` | `str \| null` | Headline string if a news event fired this step, else null |
| `position` | `dict[str, float]` | Current allocation per asset (negative = short) |
| `cash` | `float` | Available cash |
| `unrealized_pnl` | `float` | Mark-to-market PnL on open positions |
| `episode_pnl` | `float` | Cumulative realised + unrealised PnL |
| `drawdown` | `float [0, 1]` | Current drawdown from episode peak |
| `step_return` | `float` | Portfolio return on the previous step |

**What the agent must infer (not provided):**
- Current market regime (trending vs mean-reverting)
- GARCH volatility parameters for this episode
- Asset correlation structure
- News impact magnitude distribution

---

## Action Space

The agent returns a JSON object with target allocations as a fraction of total capital:

| Field | Type | Range | Tasks |
|---|---|---|---|
| `asset_A` | `float` | `[-1.0, 1.0]` | All tasks |
| `asset_B` | `float` | `[-1.0, 1.0]` | task_2, task_3 |
| `asset_C` | `float` | `[-1.0, 1.0]` | task_3 only |
| `asset_D` | `float` | `[-1.0, 1.0]` | task_3 only |
| `reasoning` | `str` | — | Optional, logged but not scored |

Positive allocation = long position (profits if price rises).  
Negative allocation = short position (profits if price falls).  
`0.0` = flat (no position).

**Environment-enforced constraints:**
- Each allocation clipped to `[-1.0, 1.0]`
- Sum of absolute allocations clipped to `1.5` (leverage limit)
- Position changes incur transaction costs of 10bps per unit of allocation change

---

## Tasks

### Task 1 — Single Asset Trend Following (Easy)
- **Assets:** asset_A only
- **Regime:** Starts trending, low transition probability (stable)
- **News:** Disabled
- **Win condition:** Sharpe ratio > 0.3 over episode
- **Score:** `clip((sharpe - 0.3) / 1.5, 0.0, 1.0)`

### Task 2 — Dual Asset Regime Navigation (Medium)
- **Assets:** asset_A, asset_B with random correlation drawn each episode
- **Regime:** Guaranteed at least one regime switch per episode
- **News:** Active, avg one event per 10 steps
- **Win condition:** Sharpe > 0.6, max drawdown < 15%
- **Score:** `clip((sharpe - 0.6) / 1.5, 0.0, 1.0)` with 50% penalty if drawdown > 15%

### Task 3 — Portfolio Under Stress (Hard)
- **Assets:** asset_A, asset_B, asset_C, asset_D in two correlated pairs
- **Regime:** Starts in high-volatility mean-reverting regime
- **News:** Frequent, avg one event per 6 steps
- **Hard constraint:** Episode terminates immediately with score 0.0 if drawdown exceeds 20%
- **Win condition:** Sharpe > 1.0, survive full 60 steps
- **Score:** `clip((sharpe - 1.0) / 1.5, 0.0, 1.0)`, zero on early termination

---

## Market Engine

### GARCH(1,1) Volatility
Volatility clusters: large moves follow large moves. Parameters are jittered ±20% each episode to vary market character.

```
variance_t = omega + alpha * return_{t-1}^2 + beta * variance_{t-1}
return_t ~ N(drift + momentum * return_{t-1}, sqrt(variance_t))
```

### Hidden Markov Regime Switching
Two hidden regimes the agent must infer from price behaviour:
- **Trending:** positive drift, strong momentum autocorrelation
- **Mean-reverting:** zero drift, negative autocorrelation (overshoots correct)

### News Shock System
Poisson process with headlines drawn from a library of 50+ templates. Impact direction is consistent per headline category; magnitude is stochastic and decays over 3 steps (100% → 60% → 30%).

### Asset Correlations
Drawn fresh each episode via Cholesky decomposition. The agent must estimate correlation from observed price co-movement — it is never provided directly.

---

## Setup & Running

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the server

```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

### Run inference (baseline agent)

```bash
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
HF_TOKEN=your_token_here \
ENV_URL=http://127.0.0.1:8000 \
python inference.py
```

### Run validation (server must be running)

```bash
ENV_URL=http://127.0.0.1:8000 python scripts/validate.py
# or let it start the server automatically:
python scripts/validate.py --start-server
```

### Build and run with Docker

```bash
docker build -t market-env .
docker run -p 8000:8000 market-env
```

---

## Project Structure

```
market-env/
├── inference.py          # Baseline agent (root dir, required by spec)
├── openenv.yaml          # Environment metadata and task definitions
├── Dockerfile
├── requirements.txt
├── README.md
├── src/
│   ├── server.py         # FastAPI HTTP server (/reset, /step, /state)
│   ├── environment.py    # MarketEnvironment class, portfolio mechanics
│   └── market_engine.py  # GARCH + HMM price generation, news system
└── scripts/
    └── validate.py       # Pre-submission validation (10 checks)
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/ping` | Health check — returns 200 |
| `GET` | `/tasks` | List all tasks with metadata |
| `POST` | `/reset` | Start new episode. Body: `{"task_id": "task_1", "seed": 42}` |
| `POST` | `/step` | Take action. Body: `{"action": {"asset_A": 0.5}}` |
| `GET` | `/state` | Get current state without advancing |

---

## Reproducibility

Pass `seed=42` on reset for the baseline evaluation run. All random processes (regime initialisation, GARCH parameter jitter, news firing, correlation sampling) are seeded from this value, producing identical scores across runs.