import numpy as np


class CancellationFeatureExtractor:
    """
    Features for cancellation prediction:
    0  - Wait time so far (minutes)
    1  - ETA of assigned driver (minutes)
    2  - Time of day (hour, 0-23)
    3  - Day of week (0=Mon … 6=Sun)
    4  - Trip distance (km)
    5  - Passenger tier (1=Standard / 2=Plus / 3=Premium)
    6  - Historical cancellation rate of this passenger (0.0-1.0; 0 if unknown)
    7  - Weather indicator (0=clear / 1=adverse)
    8  - Surge multiplier (e.g. 1.0, 1.5, 2.0)
    """

    FEATURE_NAMES = [
        'wait_time_mins',
        'driver_eta_mins',
        'hour_of_day',
        'day_of_week',
        'trip_distance_km',
        'passenger_tier',
        'historical_cancel_rate',
        'weather_indicator',
        'surge_multiplier',
    ]
    FEATURE_DIM = len(FEATURE_NAMES)

    def extract(self, trip_context: dict) -> np.ndarray:
        """
        Build a feature vector from trip_context dict.

        Expected keys (all optional — missing values default to 0):
            wait_time_mins, driver_eta_mins, hour_of_day, day_of_week,
            trip_distance_km, passenger_tier, historical_cancel_rate,
            weather_indicator, surge_multiplier

        Returns
        -------
        np.ndarray of shape (FEATURE_DIM,) with dtype float64.
        """
        features = np.array([
            float(trip_context.get('wait_time_mins', 0.0)),
            float(trip_context.get('driver_eta_mins', 0.0)),
            float(trip_context.get('hour_of_day', 0.0)),
            float(trip_context.get('day_of_week', 0.0)),
            float(trip_context.get('trip_distance_km', 0.0)),
            float(trip_context.get('passenger_tier', 1.0)),
            float(trip_context.get('historical_cancel_rate', 0.0)),
            float(trip_context.get('weather_indicator', 0.0)),
            float(trip_context.get('surge_multiplier', 1.0)),
        ], dtype=np.float64)
        return features


class LogisticCancellationModel:
    """
    Logistic regression: P(cancel | features) = σ(w^T · φ(x) + b).

    Trained on historical data via mini-batch gradient descent.

    Parameters
    ----------
    feature_dim : int   — dimensionality of feature vector (default 9)
    lr          : float — learning rate (default 0.01)
    reg_lambda  : float — L2 regularisation strength (default 0.01)
    """

    def __init__(self, feature_dim: int = 9,
                 lr: float = 0.01,
                 reg_lambda: float = 0.01):
        self.feature_dim = feature_dim
        self.lr = lr
        self.reg_lambda = reg_lambda
        # Initialise weights and bias to zero
        self.w = np.zeros(feature_dim, dtype=np.float64)
        self.b = 0.0

    # ------------------------------------------------------------------
    # Core model
    # ------------------------------------------------------------------

    def sigmoid(self, z) -> np.ndarray:
        """Numerically stable sigmoid σ(z) = 1 / (1 + exp(-z))."""
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z)),
        )

    def predict_proba(self, features) -> float:
        """
        Return P(cancel) ∈ [0, 1] for a single feature vector.

        Parameters
        ----------
        features : array-like of shape (feature_dim,)
        """
        x = np.asarray(features, dtype=np.float64)
        z = np.dot(self.w, x) + self.b
        return float(self.sigmoid(z))

    def predict(self, features, threshold: float = 0.35) -> bool:
        """
        Return True (will cancel) if P(cancel) ≥ threshold.

        Parameters
        ----------
        features  : array-like of shape (feature_dim,)
        threshold : decision boundary (default 0.35 — skewed for recall)
        """
        return self.predict_proba(features) >= threshold

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y, epochs: int = 100) -> list:
        """
        Fit model using full-batch gradient descent.

        Parameters
        ----------
        X      : array-like of shape (N, feature_dim)
        y      : array-like of shape (N,) with binary labels (0/1)
        epochs : number of passes over the training data

        Returns
        -------
        loss_history : list[float] — binary cross-entropy loss per epoch
            Loss = -(1/N) Σ [y·log(p) + (1-y)·log(1-p)] + (λ/2)·||w||²

        Gradient update:
            dL/dw = X^T·(σ(Xw) - y) / N + λ·w
            dL/db = mean(σ(Xw) - y)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        N = X.shape[0]
        eps = 1e-15  # clip to avoid log(0)

        loss_history = []

        for _ in range(epochs):
            z = X @ self.w + self.b               # (N,)
            p = self.sigmoid(z)                   # (N,)

            # Binary cross-entropy + L2
            p_clipped = np.clip(p, eps, 1 - eps)
            bce = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
            l2 = (self.reg_lambda / 2.0) * np.dot(self.w, self.w)
            loss_history.append(float(bce + l2))

            # Gradients
            error = p - y                         # (N,)
            dw = (X.T @ error) / N + self.reg_lambda * self.w
            db = float(np.mean(error))

            # Parameter update
            self.w -= self.lr * dw
            self.b -= self.lr * db

        return loss_history

    def update_online(self, feature, label):
        """
        Single-sample online (stochastic) gradient update.

        Parameters
        ----------
        feature : array-like of shape (feature_dim,)
        label   : int/float — 1 if cancelled, 0 otherwise
        """
        x = np.asarray(feature, dtype=np.float64)
        y = float(label)
        z = np.dot(self.w, x) + self.b
        p = float(self.sigmoid(z))
        error = p - y

        self.w -= self.lr * (error * x + self.reg_lambda * self.w)
        self.b -= self.lr * error


class NoShowRiskManager:
    """
    Uses LogisticCancellationModel to proactively manage no-show risk.

    Parameters
    ----------
    model                   : LogisticCancellationModel instance
    high_risk_threshold     : P(cancel) above which a trip is 'high' risk (default 0.6)
    reassign_lead_time_mins : how far ahead to act on high-risk trips (default 3.0)
    """

    RISK_LEVELS = {
        'low': (0.0, 0.35),
        'medium': (0.35, 0.60),
        'high': (0.60, 1.01),
    }

    def __init__(self, model: LogisticCancellationModel,
                 high_risk_threshold: float = 0.6,
                 reassign_lead_time_mins: float = 3.0):
        self.model = model
        self.high_risk_threshold = high_risk_threshold
        self.reassign_lead_time_mins = reassign_lead_time_mins
        self._extractor = CancellationFeatureExtractor()

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------

    def assess_risk(self, active_trips: list) -> list:
        """
        Assess cancellation risk for each active trip.

        Parameters
        ----------
        active_trips : list of trip_context dicts, each must have 'trip_id' plus
                       any features recognised by CancellationFeatureExtractor.

        Returns
        -------
        list of dicts: [{'trip_id', 'cancel_proba', 'risk_level'}]
        """
        assessments = []
        for ctx in active_trips:
            features = self._extractor.extract(ctx)
            proba = self.model.predict_proba(features)
            level = self._classify_risk(proba)
            assessments.append({
                'trip_id': ctx.get('trip_id', 'unknown'),
                'cancel_proba': proba,
                'risk_level': level,
            })
        return assessments

    def _classify_risk(self, proba: float) -> str:
        for level, (lo, hi) in self.RISK_LEVELS.items():
            if lo <= proba < hi:
                return level
        return 'high'

    # ------------------------------------------------------------------
    # Action suggestions
    # ------------------------------------------------------------------

    def suggest_preemptive_actions(self, risk_assessments: list) -> list:
        """
        Suggest proactive actions based on risk assessments.

        Actions by risk level:
        - 'low'    : no action
        - 'medium' : send push notification to confirm trip
        - 'high'   : pre-dispatch backup driver (if driver ETA >
                     reassign_lead_time_mins), send confirmation SMS,
                     consider discount incentive

        Parameters
        ----------
        risk_assessments : output of assess_risk()

        Returns
        -------
        list of action dicts:
            {'trip_id', 'risk_level', 'cancel_proba', 'actions': [str]}
        """
        suggestions = []
        for assessment in risk_assessments:
            trip_id = assessment['trip_id']
            proba = assessment['cancel_proba']
            level = assessment['risk_level']

            actions = []
            if level == 'low':
                actions = []
            elif level == 'medium':
                actions = [
                    'send_push_notification: Please confirm your upcoming ride.',
                ]
            else:  # high
                actions = [
                    'pre_dispatch_backup_driver: Assign standby vehicle now.',
                    'send_sms_confirmation: Urgent confirmation request to passenger.',
                    'offer_discount_incentive: Apply 10% discount to reduce cancellation.',
                ]
                if proba >= 0.80:
                    actions.append(
                        'flag_for_manual_review: Very high risk — ops team to monitor.'
                    )

            suggestions.append({
                'trip_id': trip_id,
                'risk_level': level,
                'cancel_proba': proba,
                'actions': actions,
            })

        return suggestions
