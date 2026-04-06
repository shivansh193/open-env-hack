"""
market_engine.py
----------------
Core price generation engine for the Synthetic Market RL Environment.

Implements:
  - HMM regime switching (trending / mean-reverting)
  - GARCH(1,1) volatility clustering
  - Correlated multi-asset returns via Cholesky decomposition
  - News shock system (Poisson process)
  - Volatility signal normalization
"""

import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Regime parameters
# ---------------------------------------------------------------------------

REGIMES = {
    "trending": {
        "drift": 0.0008,
        "momentum": 0.6,
        "transition_prob": 0.05,  # prob of switching to mean_reverting each step
    },
    "mean_reverting": {
        "drift": 0.0,
        "momentum": -0.3,
        "transition_prob": 0.08,  # prob of switching to trending each step
    },
}

REGIME_NAMES = list(REGIMES.keys())  # ["trending", "mean_reverting"]

# ---------------------------------------------------------------------------
# GARCH(1,1) base parameters
# ---------------------------------------------------------------------------

GARCH_BASE = {
    "omega": 0.000002,
    "alpha": 0.08,   # ARCH term — shock sensitivity
    "beta":  0.90,   # GARCH term — vol persistence
}

# ---------------------------------------------------------------------------
# News headline library (50+ templates)
# ---------------------------------------------------------------------------

NEWS_TEMPLATES = [
    # (template, direction_map, magnitude_range)
    # direction_map: +1 positive, -1 negative, 0 no effect, per asset key
    # magnitude_range: (low, high) uniform draw

    # --- Macro ---
    ("Central authority signals tightening of monetary conditions",
     {"A": -1, "B": -1, "C": -1, "D": -1}, (0.010, 0.030)),

    ("Regional output data exceeds consensus projections by {pct}%",
     {"A": +1, "B": +1, "C": +1, "D": +1}, (0.008, 0.025)),

    ("Inflation indicators rise above target threshold in {region}",
     {"A": -1, "B": -1, "C":  0, "D":  0}, (0.006, 0.018)),

    ("Central authority holds policy rate steady, guidance dovish",
     {"A": +1, "B": +1, "C": +1, "D": +1}, (0.005, 0.015)),

    ("Regional GDP growth revised downward for {region} bloc",
     {"A": -1, "B": -1, "C": -1, "D": -1}, (0.008, 0.022)),

    ("Employment figures for {region} region disappoint expectations",
     {"A": -1, "B":  0, "C": -1, "D":  0}, (0.005, 0.014)),

    ("Consumer confidence index reaches multi-year high in {region}",
     {"A": +1, "B": +1, "C":  0, "D":  0}, (0.006, 0.016)),

    ("Trade surplus widens for {region} bloc, currency strengthens",
     {"A": +1, "B": -1, "C": +1, "D": -1}, (0.007, 0.018)),

    # --- Sector ---
    ("Quarterly output for {sector} sector exceeds projections by {pct}%",
     {"A": +1, "B": +1, "C":  0, "D":  0}, (0.008, 0.025)),

    ("Supply chain disruptions reported in {region} manufacturing sector",
     {"A": +1, "B":  0, "C": +1, "D":  0}, (0.005, 0.020)),

    ("Regulatory body announces investigation into {sector} pricing practices",
     {"A": -1, "B":  0, "C": -1, "D":  0}, (0.010, 0.025)),

    ("{sector} sector demand forecasts cut by major research house",
     {"A": -1, "B": -1, "C":  0, "D":  0}, (0.008, 0.020)),

    ("New capacity constraints emerge in {sector} supply chain",
     {"A": +1, "B":  0, "C":  0, "D": +1}, (0.006, 0.018)),

    ("Productivity gains in {sector} sector exceed analyst expectations",
     {"A": +1, "B": +1, "C":  0, "D":  0}, (0.007, 0.019)),

    ("Merger activity reported in {sector} sector, consolidation expected",
     {"A": +1, "B": +1, "C":  0, "D":  0}, (0.010, 0.028)),

    ("Overcapacity concerns weigh on {sector} sector outlook",
     {"A": -1, "B": -1, "C":  0, "D":  0}, (0.006, 0.016)),

    ("New regulatory framework proposed for {sector} operators",
     {"A": -1, "B":  0, "C": -1, "D":  0}, (0.008, 0.022)),

    ("Commodity input costs surge for {sector} manufacturers",
     {"A": -1, "B":  0, "C": -1, "D": +1}, (0.009, 0.024)),

    # --- Geopolitical ---
    ("Geopolitical tensions in {region} escalate, risk-off sentiment dominates",
     {"A": -1, "B": +1, "C": -1, "D": +1}, (0.015, 0.035)),

    ("Peace negotiations resume between {region} bloc factions",
     {"A": +1, "B": -1, "C": +1, "D": -1}, (0.010, 0.025)),

    ("Sanctions regime expanded targeting {region} export sector",
     {"A": -1, "B":  0, "C": -1, "D":  0}, (0.012, 0.030)),

    ("Bilateral trade agreement signed between {region} blocs",
     {"A": +1, "B": +1, "C":  0, "D":  0}, (0.008, 0.022)),

    ("Political transition in {region} raises policy uncertainty",
     {"A": -1, "B":  0, "C": -1, "D":  0}, (0.010, 0.028)),

    ("Regional authority imposes export controls on {sector} goods",
     {"A": -1, "B": +1, "C": -1, "D":  0}, (0.012, 0.030)),

    ("Ceasefire agreement reduces risk premium in {region} assets",
     {"A": +1, "B": -1, "C": +1, "D":  0}, (0.010, 0.024)),

    # --- Commodity / Resource ---
    ("Energy commodity prices spike following {region} supply disruption",
     {"A": -1, "B":  0, "C": -1, "D": +1}, (0.012, 0.032)),

    ("Agricultural output in {region} improves on favorable conditions",
     {"A": +1, "B":  0, "C": +1, "D":  0}, (0.005, 0.015)),

    ("Industrial metal inventories fall below seasonal norms",
     {"A": +1, "B":  0, "C": +1, "D":  0}, (0.006, 0.018)),

    ("Energy transition investment accelerates in {region}",
     {"A": -1, "B": +1, "C": -1, "D": +1}, (0.007, 0.020)),

    ("Commodity cartel signals production cut of {pct}%",
     {"A": -1, "B":  0, "C": +1, "D": +1}, (0.010, 0.028)),

    # --- Financial / Credit ---
    ("Credit spreads widen as default risk rises in {sector} sector",
     {"A": -1, "B": -1, "C": -1, "D": -1}, (0.010, 0.025)),

    ("Institutional demand for risk assets surges on improved sentiment",
     {"A": +1, "B": +1, "C": +1, "D": +1}, (0.008, 0.022)),

    ("Liquidity conditions tighten in interbank market",
     {"A": -1, "B": -1, "C":  0, "D":  0}, (0.007, 0.018)),

    ("Capital inflows to {region} assets accelerate",
     {"A": +1, "B": +1, "C":  0, "D":  0}, (0.006, 0.016)),

    ("Short interest in {sector} sector reaches elevated levels",
     {"A": -1, "B":  0, "C": -1, "D":  0}, (0.008, 0.020)),

    ("Risk appetite recovers following central authority reassurance",
     {"A": +1, "B": +1, "C": +1, "D": +1}, (0.007, 0.020)),

    # --- Technology / Innovation ---
    ("Breakthrough announced in {sector} efficiency technology",
     {"A": +1, "B": +1, "C":  0, "D":  0}, (0.010, 0.028)),

    ("Cybersecurity incident disrupts {sector} infrastructure in {region}",
     {"A": -1, "B":  0, "C": -1, "D":  0}, (0.008, 0.022)),

    ("Automation deployment reduces cost base in {sector} operations",
     {"A": +1, "B":  0, "C": +1, "D":  0}, (0.006, 0.016)),

    ("Patent dispute halts production for {sector} operators",
     {"A": -1, "B":  0, "C":  0, "D":  0}, (0.007, 0.018)),

    # --- Climate / Natural ---
    ("Severe weather disrupts logistics in {region}, delays shipments",
     {"A": -1, "B":  0, "C": -1, "D": +1}, (0.008, 0.020)),

    ("Climate accord signed, long-term {sector} demand outlook shifts",
     {"A": -1, "B": +1, "C": -1, "D": +1}, (0.010, 0.025)),

    ("Drought conditions threaten agricultural output in {region}",
     {"A": +1, "B":  0, "C": +1, "D":  0}, (0.007, 0.019)),

    # --- Sentiment / Survey ---
    ("Business sentiment survey shows sharp deterioration in {region}",
     {"A": -1, "B": -1, "C":  0, "D":  0}, (0.006, 0.016)),

    ("Analyst consensus upgrades {sector} sector outlook to overweight",
     {"A": +1, "B": +1, "C":  0, "D":  0}, (0.008, 0.022)),

    ("Retail flow data shows rotation out of {sector} into safety assets",
     {"A": -1, "B": +1, "C": -1, "D": +1}, (0.006, 0.016)),

    ("Survey indicates supply bottleneck easing in {region}",
     {"A": +1, "B":  0, "C": +1, "D":  0}, (0.005, 0.014)),

    # --- Idiosyncratic ---
    ("Regional Output Index for {region} revised significantly upward",
     {"A": +1, "B":  0, "C": +1, "D":  0}, (0.010, 0.026)),

    ("Sector A reports unexpected inventory build, demand concerns rise",
     {"A": -1, "B":  0, "C":  0, "D":  0}, (0.008, 0.020)),

    ("Commodity B supply glut emerges as producers ramp capacity",
     {"A":  0, "B": -1, "C":  0, "D": -1}, (0.009, 0.022)),

    ("Cross-sector contagion fears emerge following {sector} writedowns",
     {"A": -1, "B": -1, "C": -1, "D": -1}, (0.012, 0.030)),
]

# Template variable fill lists
_REGIONS  = ["Northern", "Southern", "Eastern", "Western", "Central", "Pacific", "Atlantic"]
_SECTORS  = ["Sector A", "Sector B", "Sector C", "Industrial", "Consumer", "Financial", "Energy"]
_PCTS     = ["2", "3", "5", "7", "8", "10", "12", "15"]


# ---------------------------------------------------------------------------
# MarketEngine
# ---------------------------------------------------------------------------

class MarketEngine:
    """
    Generates synthetic market data for one episode.

    Parameters
    ----------
    n_assets : int
        Number of assets (1, 2, or 4).
    task_config : dict
        Task-level overrides (see tasks.py for examples):
          - initial_regime: "trending" | "mean_reverting" | None (random)
          - transition_scale: float multiplier on transition probs (default 1.0)
          - news_lambda: float Poisson rate per step (default 0.1)
          - garch_scale: float multiplier on GARCH omega/alpha (default 1.0)
          - force_regime_switch: bool — guarantee a switch in task 2
    seed : int | None
        RNG seed for reproducibility.
    """

    ASSET_KEYS = ["A", "B", "C", "D"]

    def __init__(
        self,
        n_assets: int = 1,
        task_config: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        assert n_assets in (1, 2, 4), "n_assets must be 1, 2, or 4"
        self.n_assets    = n_assets
        self.asset_keys  = self.ASSET_KEYS[:n_assets]
        self.task_config = task_config or {}
        self.seed        = seed

        self.rng: np.random.RandomState = None  # set in reset()

        # Episode state (populated by reset())
        self.current_regime:  str             = None
        self.garch_params:    dict            = None
        self.corr_matrix:     np.ndarray      = None
        self.chol:            np.ndarray      = None
        self.prices:          np.ndarray      = None  # shape (n_assets,)
        self.prev_returns:    np.ndarray      = None  # shape (n_assets,)
        self.variances:       np.ndarray      = None  # shape (n_assets,)
        self.price_history:   list            = None  # list of np.ndarray
        self.vol_history:     list            = None  # list of floats (avg sigma)
        self.news_decay:      list            = None  # active news impacts
        self._regime_switched: bool           = False
        self._step:           int             = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """
        Initialize a new episode. Returns the first price snapshot.
        """
        seed = self.seed if self.seed is not None else np.random.randint(0, 99999)
        self.rng = np.random.RandomState(seed)

        # Initial regime
        init = self.task_config.get("initial_regime", None)
        if init is not None:
            self.current_regime = init
        else:
            self.current_regime = self.rng.choice(REGIME_NAMES)

        # GARCH params — jitter ±20%
        scale = self.task_config.get("garch_scale", 1.0)
        jitter = lambda v: v * (1 + self.rng.uniform(-0.20, 0.20)) * scale
        self.garch_params = {k: jitter(v) for k, v in GARCH_BASE.items()}
        # Clamp for stability: alpha + beta < 1
        a, b = self.garch_params["alpha"], self.garch_params["beta"]
        if a + b >= 1.0:
            self.garch_params["beta"] = 0.98 - a

        # Initial variance at long-run mean
        om, al, be = (
            self.garch_params["omega"],
            self.garch_params["alpha"],
            self.garch_params["beta"],
        )
        var0 = om / (1.0 - al - be)
        self.variances    = np.full(self.n_assets, var0)
        self.prev_returns = np.zeros(self.n_assets)

        # Correlation matrix + Cholesky
        self.corr_matrix = self._draw_correlation()
        self.chol        = np.linalg.cholesky(self.corr_matrix)

        # Starting prices ~ 100 with small dispersion
        self.prices = self.rng.uniform(95.0, 105.0, size=self.n_assets)

        # History buffers (pre-fill with t=0 price)
        self.price_history = [self.prices.copy() for _ in range(10)]
        self.vol_history   = [float(np.sqrt(var0))] * 10
        self.news_decay    = []   # list of {asset_impacts, steps_left}
        self._regime_switched = False
        self._step = 0

        return self._snapshot()

    def step(self) -> dict:
        """
        Advance the market by one step. Returns updated price snapshot.
        Must call reset() before step().
        """
        assert self.prices is not None, "Call reset() before step()"

        self._step += 1

        # 1. Possibly switch regime
        self._maybe_switch_regime()

        # 2. Draw correlated standard normals
        z_indep   = self.rng.standard_normal(self.n_assets)
        z_corr    = self.chol @ z_indep  # shape (n_assets,)

        # 3. Compute returns per asset via GARCH + HMM
        new_returns = self._compute_returns(z_corr)

        # 4. Apply news shocks
        news_event = self._process_news(new_returns)

        # 5. Update prices
        self.prices = self.prices * (1.0 + new_returns)
        self.prices = np.maximum(self.prices, 0.01)  # floor at penny

        # 6. Update GARCH variances for next step
        om, al, be = (
            self.garch_params["omega"],
            self.garch_params["alpha"],
            self.garch_params["beta"],
        )
        self.variances = om + al * new_returns**2 + be * self.variances
        self.variances = np.maximum(self.variances, 1e-8)

        # 7. Store history
        self.prev_returns = new_returns.copy()
        self.price_history.append(self.prices.copy())
        avg_sigma = float(np.mean(np.sqrt(self.variances)))
        self.vol_history.append(avg_sigma)

        return self._snapshot(news_event=news_event)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_returns(self, z_corr: np.ndarray) -> np.ndarray:
        """
        Compute asset returns using GARCH(1,1) + HMM regime parameters.
        """
        regime = REGIMES[self.current_regime]
        drift    = regime["drift"]
        momentum = regime["momentum"]
        returns  = np.empty(self.n_assets)

        for i in range(self.n_assets):
            sigma_t   = np.sqrt(self.variances[i])
            mu_t      = drift + momentum * self.prev_returns[i]
            returns[i] = mu_t + sigma_t * z_corr[i]

        return returns

    def _maybe_switch_regime(self):
        """
        Sample HMM transition for current step.
        For task 2: guarantee at least one switch by step 45 if none yet.
        """
        regime   = REGIMES[self.current_regime]
        t_scale  = self.task_config.get("transition_scale", 1.0)
        t_prob   = regime["transition_prob"] * t_scale

        # Force switch logic for Task 2
        force = self.task_config.get("force_regime_switch", False)
        if force and not self._regime_switched and self._step >= 45:
            t_prob = 1.0

        if self.rng.random() < t_prob:
            # Switch to the other regime
            other = [r for r in REGIME_NAMES if r != self.current_regime][0]
            self.current_regime   = other
            self._regime_switched = True

    def _process_news(self, returns: np.ndarray) -> Optional[str]:
        """
        1. Apply decay from prior news events.
        2. Possibly fire a new Poisson news event.
        Returns headline string or None.
        """
        lam = self.task_config.get("news_lambda", 0.0)
        headline = None

        # Apply existing decay impacts
        still_active = []
        for shock in self.news_decay:
            for i, key in enumerate(self.asset_keys):
                returns[i] += shock["impacts"].get(key, 0.0)
            shock["steps_left"] -= 1
            if shock["steps_left"] > 0:
                still_active.append(shock)
        self.news_decay = still_active

        # Fire new event?
        if lam > 0 and self.rng.random() < lam:
            idx = self.rng.randint(0, len(NEWS_TEMPLATES))
            tmpl, dir_map, mag_range = NEWS_TEMPLATES[idx]

            magnitude = self.rng.uniform(*mag_range)
            headline  = self._fill_template(tmpl)

            # Build impact dict for this episode's assets
            # Full impact now, 60% step+1, 30% step+2
            impacts_now  = {}
            impacts_d1   = {}
            impacts_d2   = {}

            for i, key in enumerate(self.asset_keys):
                direction = dir_map.get(key, 0)
                if direction == 0:
                    continue
                impacts_now[key] = direction * magnitude
                impacts_d1[key]  = direction * magnitude * 0.60
                impacts_d2[key]  = direction * magnitude * 0.30

            # Apply immediate impact
            for i, key in enumerate(self.asset_keys):
                returns[i] += impacts_now.get(key, 0.0)

            # Queue decay
            if impacts_d1:
                self.news_decay.append({"impacts": impacts_d1, "steps_left": 1})
            if impacts_d2:
                self.news_decay.append({"impacts": impacts_d2, "steps_left": 2})

        return headline

    def _fill_template(self, template: str) -> str:
        """Fill {region}, {sector}, {pct} placeholders with random values."""
        result = template
        if "{region}" in result:
            result = result.replace("{region}", self.rng.choice(_REGIONS))
        if "{sector}" in result:
            result = result.replace("{sector}", self.rng.choice(_SECTORS))
        if "{pct}" in result:
            result = result.replace("{pct}", self.rng.choice(_PCTS))
        return result

    def _draw_correlation(self) -> np.ndarray:
        """
        Draw a valid positive-definite correlation matrix for self.n_assets.

        n_assets=1: [[1.0]]
        n_assets=2: [[1, r], [r, 1]], r ~ Uniform(0.3, 0.8)
        n_assets=4: two pairs with high intra-pair (0.6-0.9),
                    low cross-pair (0.0-0.3) correlation
        """
        n = self.n_assets
        if n == 1:
            return np.array([[1.0]])

        if n == 2:
            r   = self.rng.uniform(0.3, 0.8)
            mat = np.array([[1.0, r], [r, 1.0]])
            return self._nearest_pd(mat)

        # n == 4: block structure
        r_a    = self.rng.uniform(0.6, 0.9)   # Pair A internal
        r_b    = self.rng.uniform(0.6, 0.9)   # Pair B internal
        r_cross = self.rng.uniform(0.0, 0.3)  # cross-pair

        mat = np.array([
            [1.0,   r_a,    r_cross, r_cross],
            [r_a,   1.0,    r_cross, r_cross],
            [r_cross, r_cross, 1.0,  r_b    ],
            [r_cross, r_cross, r_b,  1.0    ],
        ])
        return self._nearest_pd(mat)

    @staticmethod
    def _nearest_pd(mat: np.ndarray) -> np.ndarray:
        """
        Project matrix to nearest positive-definite matrix (Higham 1988).
        Ensures Cholesky decomposition always succeeds.
        """
        sym = (mat + mat.T) / 2
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals = np.maximum(eigvals, 1e-6)
        pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Renormalize diagonal to 1 (keep it a correlation matrix)
        d = np.sqrt(np.diag(pd))
        pd = pd / np.outer(d, d)
        return pd

    def _snapshot(self, news_event: Optional[str] = None) -> dict:
        """
        Build the price/vol snapshot dict returned to the environment.
        """
        prices_out = {}
        for i, key in enumerate(self.asset_keys):
            history = [float(h[i]) for h in self.price_history[-10:]]
            prices_out[f"asset_{key}"] = history

        # Volatility signal — normalize current avg sigma to [0, 1]
        # using a rolling window of the past vol observations
        recent_vols = self.vol_history[-20:] if len(self.vol_history) >= 2 else self.vol_history
        current_vol = self.vol_history[-1]
        vol_min = min(recent_vols)
        vol_max = max(recent_vols)
        if vol_max > vol_min:
            vol_signal = float((current_vol - vol_min) / (vol_max - vol_min))
        else:
            vol_signal = 0.5

        return {
            "prices":           prices_out,
            "volatility_signal": round(vol_signal, 4),
            "news":             news_event,
            "current_regime":   self.current_regime,   # for internal use; stripped in server
            "raw_variances":    [float(v) for v in self.variances],  # for reward.py
        }

    # ------------------------------------------------------------------
    # Utility / introspection
    # ------------------------------------------------------------------

    def get_current_prices(self) -> np.ndarray:
        return self.prices.copy()

    def get_current_regime(self) -> str:
        return self.current_regime

    def get_correlation_matrix(self) -> np.ndarray:
        return self.corr_matrix.copy()

    def regime_switched(self) -> bool:
        return self._regime_switched