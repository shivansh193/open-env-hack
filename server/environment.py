"""
environment.py
--------------
MarketEnvironment — wraps MarketEngine with full portfolio state,
position management, reward calculation, and episode lifecycle.

Consumed by server.py via:
    env = MarketEnvironment(task_config)
    state = env.reset(task_id, seed)
    state, reward, done, info = env.step(action)
    state = env.get_state()
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

from .market_engine import MarketEngine


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "task_1": {
        "n_assets":           1,
        "initial_regime":     "trending",
        "transition_scale":   0.5,        # halved — more stable
        "news_lambda":        0.0,         # no news
        "garch_scale":        1.0,
        "force_regime_switch": False,
        "max_steps":          60,
        "sharpe_threshold":   0.3,         # grader lower bound
        "max_drawdown_limit": None,        # no hard drawdown limit
        "early_terminate_dd": None,
        "description":        "Single asset trend following",
    },
    "task_2": {
        "n_assets":           2,
        "initial_regime":     None,        # random start
        "transition_scale":   1.0,
        "news_lambda":        0.1,         # avg 1 event / 10 steps
        "garch_scale":        1.0,
        "force_regime_switch": True,       # guaranteed switch
        "max_steps":          60,
        "sharpe_threshold":   0.6,
        "max_drawdown_limit": 0.15,        # soft — penalised in grader
        "early_terminate_dd": None,
        "description":        "Dual asset regime navigation",
    },
    "task_3": {
        "n_assets":           4,
        "initial_regime":     "mean_reverting",
        "transition_scale":   1.0,
        "news_lambda":        1 / 6,       # avg 1 event / 6 steps
        "garch_scale":        1.5,         # volatile
        "force_regime_switch": False,
        "max_steps":          60,
        "sharpe_threshold":   1.0,
        "max_drawdown_limit": 0.20,        # hard — early termination
        "early_terminate_dd": 0.20,
        "description":        "Portfolio under stress",
    },
}

INITIAL_CAPITAL = 100_000.0
TRANSACTION_COST_RATE = 0.001   # 10bps per unit of allocation change
DRAWDOWN_PENALTY_RATE = 2.0     # multiplier when drawdown > threshold
DRAWDOWN_PENALTY_THRESHOLD = 0.10
SHARPE_ANNUALISE = np.sqrt(252)


# ---------------------------------------------------------------------------
# MarketEnvironment
# ---------------------------------------------------------------------------

class MarketEnvironment:
    """
    Full RL environment for one task.

    Action schema (dict):
        {
            "asset_A": float,   # target allocation fraction [-1, 1]
            "asset_B": float,   # tasks 2+
            "asset_C": float,   # task 3 only
            "asset_D": float,   # task 3 only
            "reasoning": str    # optional, logged but not used
        }

    Constraints enforced:
        - Each allocation clipped to [-1, 1]
        - Sum of abs(allocations) clipped to 1.5 (max leverage)
    """

    def __init__(self):
        self.engine:         MarketEngine      = None
        self.task_id:        str               = None
        self.task_cfg:       dict              = None
        self.asset_keys:     list              = None   # ["A", "B", ...]

        # Portfolio state
        self.cash:           float             = INITIAL_CAPITAL
        self.shares:         Dict[str, float]  = {}
        self.allocations:    Dict[str, float]  = {}     # current target allocs
        self.initial_capital: float            = INITIAL_CAPITAL

        # Episode tracking
        self.step_count:     int               = 0
        self.portfolio_value: float            = INITIAL_CAPITAL
        self.peak_value:     float             = INITIAL_CAPITAL
        self.step_returns:   list              = []
        self.episode_done:   bool              = False
        self.early_terminated: bool            = False

        # Latest market snapshot from engine
        self._market_snap:   dict              = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task_1", seed: Optional[int] = None) -> dict:
        """
        Start a new episode for the given task.
        Returns the initial state dict (no position yet).
        """
        assert task_id in TASK_CONFIGS, f"Unknown task_id: {task_id}"

        self.task_id  = task_id
        self.task_cfg = TASK_CONFIGS[task_id].copy()

        # Build and reset engine
        engine_cfg = {k: v for k, v in self.task_cfg.items()
                      if k not in ("max_steps", "sharpe_threshold",
                                   "max_drawdown_limit", "early_terminate_dd",
                                   "description", "n_assets")}
        self.engine = MarketEngine(
            n_assets=self.task_cfg["n_assets"],
            task_config=engine_cfg,
            seed=seed,
        )
        self._market_snap = self.engine.reset()
        self.asset_keys   = self.engine.asset_keys

        # Reset portfolio
        self.cash             = INITIAL_CAPITAL
        self.initial_capital  = INITIAL_CAPITAL
        self.shares           = {k: 0.0 for k in self.asset_keys}
        self.allocations      = {k: 0.0 for k in self.asset_keys}
        self.portfolio_value  = INITIAL_CAPITAL
        self.peak_value       = INITIAL_CAPITAL
        self.step_count       = 0
        self.step_returns     = []
        self.episode_done     = False
        self.early_terminated = False

        return self._build_state()

    def step(self, action: dict) -> Tuple[dict, float, bool, dict]:
        """
        Apply action, advance market, compute reward.

        Returns
        -------
        state  : dict
        reward : float
        done   : bool
        info   : dict  — includes grader_score on final step
        """
        assert not self.episode_done, "Episode is done. Call reset()."
        assert self._market_snap is not None, "Call reset() first."

        self.step_count += 1

        # 1. Parse and validate action
        target_allocs = self._parse_action(action)

        # 2. Record portfolio value before trade
        prev_value = self.portfolio_value

        # 3. Execute trades (update shares/cash, apply transaction costs)
        tx_cost = self._execute_trades(target_allocs)

        # 4. Advance market
        self._market_snap = self.engine.step()

        # 5. Mark to market
        self._mark_to_market()

        # 6. Compute step return
        step_ret = (self.portfolio_value - prev_value) / self.initial_capital
        self.step_returns.append(step_ret)

        # 7. Update peak / drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        drawdown = self._compute_drawdown()

        # 8. Reward
        dd_penalty = DRAWDOWN_PENALTY_RATE * max(0.0, drawdown - DRAWDOWN_PENALTY_THRESHOLD)
        reward     = step_ret - tx_cost - dd_penalty

        # 9. Done?
        max_steps = self.task_cfg["max_steps"]
        early_dd  = self.task_cfg.get("early_terminate_dd")
        done      = False

        if early_dd is not None and drawdown > early_dd:
            done                  = True
            self.early_terminated = True

        if self.step_count >= max_steps:
            done = True

        self.episode_done = done

        # 10. Info
        info = {
            "step":             self.step_count,
            "drawdown":         round(drawdown, 6),
            "max_drawdown":     round(self._max_drawdown(), 6),
            "portfolio_value":  round(self.portfolio_value, 4),
            "transaction_cost": round(tx_cost, 6),
            "early_terminated": self.early_terminated,
        }

        if done:
            info["grader_score"] = self._grade()
            info["sharpe"]       = round(self._compute_sharpe(), 4)

        state = self._build_state()
        return state, round(reward, 8), done, info

    def get_state(self) -> dict:
        """Return current state without advancing the environment."""
        assert self._market_snap is not None, "Call reset() first."
        return self._build_state()

    # ------------------------------------------------------------------
    # Portfolio mechanics
    # ------------------------------------------------------------------

    def _parse_action(self, action: dict) -> Dict[str, float]:
        """
        Extract per-asset allocations, clip, and normalise leverage.
        """
        allocs = {}
        for key in self.asset_keys:
            raw = action.get(f"asset_{key}", 0.0)
            try:
                raw = float(raw)
            except (TypeError, ValueError):
                raw = 0.0
            allocs[key] = np.clip(raw, -1.0, 1.0)

        # Enforce max leverage 1.5
        total_abs = sum(abs(v) for v in allocs.values())
        if total_abs > 1.5:
            scale = 1.5 / total_abs
            allocs = {k: v * scale for k, v in allocs.items()}

        return allocs

    def _execute_trades(self, target_allocs: Dict[str, float]) -> float:
        """
        Rebalance portfolio to target allocations.
        Transaction cost = 0.001 * sum(|delta_alloc|).
        Returns total transaction cost (as fraction of initial capital).
        """
        prices = self._current_prices()

        # Compute allocation change
        alloc_change = sum(
            abs(target_allocs.get(k, 0.0) - self.allocations.get(k, 0.0))
            for k in self.asset_keys
        )
        tx_cost = TRANSACTION_COST_RATE * alloc_change

        # Rebalance: convert allocations to shares
        total_capital = self.portfolio_value
        for key in self.asset_keys:
            target_value = target_allocs[key] * total_capital
            target_shares = target_value / prices[key] if prices[key] > 0 else 0.0
            self.shares[key] = target_shares

        # Cash = uninvested portion
        invested = sum(
            self.shares[k] * prices[k]
            for k in self.asset_keys
        )
        self.cash = total_capital - invested - (tx_cost * self.initial_capital)
        self.allocations = target_allocs.copy()

        return tx_cost

    def _mark_to_market(self):
        """Update portfolio_value from current prices and shares."""
        prices = self._current_prices()
        holdings_value = sum(self.shares[k] * prices[k] for k in self.asset_keys)
        self.portfolio_value = self.cash + holdings_value

    def _current_prices(self) -> Dict[str, float]:
        """Latest price for each asset (last element of 10-step history)."""
        return {
            key: self._market_snap["prices"][f"asset_{key}"][-1]
            for key in self.asset_keys
        }

    def _compute_drawdown(self) -> float:
        """Current drawdown from episode peak."""
        if self.peak_value <= 0:
            return 0.0
        dd = (self.peak_value - self.portfolio_value) / self.peak_value
        return max(0.0, dd)

    def _max_drawdown(self) -> float:
        """
        Max drawdown over episode using step_returns.
        Reconstructs cumulative portfolio path.
        """
        if not self.step_returns:
            return 0.0
        cum = np.cumprod(1 + np.array(self.step_returns))
        rolling_max = np.maximum.accumulate(cum)
        dd_series = (rolling_max - cum) / rolling_max
        return float(np.max(dd_series))

    # ------------------------------------------------------------------
    # State builder
    # ------------------------------------------------------------------

    def _build_state(self) -> dict:
        """
        Assemble the full state dict sent to the agent.
        Strips internal fields (current_regime, raw_variances) from market snap.
        """
        snap    = self._market_snap
        prices  = self._current_prices()

        unrealized_pnl = sum(
            self.shares[k] * prices[k]
            for k in self.asset_keys
        ) - (self.portfolio_value - self.cash)

        episode_pnl = self.portfolio_value - self.initial_capital

        # Volume: synthetic — proportional to volatility signal with noise
        # Gives agent a correlated but noisy signal
        vol_sig   = snap["volatility_signal"]
        base_vol  = int(1_000_000 * (0.5 + vol_sig))
        rng_state = self.engine.rng  # reuse engine RNG for consistency
        volume    = {
            f"asset_{k}": [
                int(base_vol * rng_state.uniform(0.7, 1.3))
                for _ in range(10)
            ]
            for k in self.asset_keys
        }

        state = {
            "step":             self.step_count,
            "prices":           snap["prices"],
            "volume":           volume,
            "volatility_signal": snap["volatility_signal"],
            "news":             snap.get("news"),
            "position":         {f"asset_{k}": round(self.shares[k], 6)
                                 for k in self.asset_keys},
            "cash":             round(self.cash, 4),
            "unrealized_pnl":   round(unrealized_pnl, 4),
            "episode_pnl":      round(episode_pnl, 4),
            "drawdown":         round(self._compute_drawdown(), 6),
            "step_return":      round(self.step_returns[-1], 8)
                                if self.step_returns else 0.0,
        }
        return state

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    def _compute_sharpe(self) -> float:
        r = np.array(self.step_returns)
        if len(r) < 2 or np.std(r) == 0:
            return 0.0
        return float(np.mean(r) / np.std(r) * SHARPE_ANNUALISE)

    def _grade(self) -> float:
        """
        Compute 0.0–1.0 grader score for the completed episode.
        """
        returns = self.step_returns
        if len(returns) < 2:
            return 0.0

        sharpe  = self._compute_sharpe()
        max_dd  = self._max_drawdown()
        cfg     = self.task_cfg
        thresh  = cfg["sharpe_threshold"]

        if self.task_id == "task_1":
            score = float(np.clip((sharpe - thresh) / 1.5, 0.0, 1.0))

        elif self.task_id == "task_2":
            score = float(np.clip((sharpe - thresh) / 1.5, 0.0, 1.0))
            if max_dd > 0.15:
                score *= 0.5    # penalise but don't zero

        elif self.task_id == "task_3":
            if self.early_terminated:
                return 0.0      # hard fail
            score = float(np.clip((sharpe - thresh) / 1.5, 0.0, 1.0))

        else:
            score = 0.0

        # Episode-end Sharpe bonus absorbed into score (already computed above)
        return round(score, 6)