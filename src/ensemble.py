"""Reusable ensemble model definition with a stable import path for joblib."""

import numpy as np


class EnsembleModel:
    """Average multiple tree models, with an optional RandomForest member."""

    def __init__(
        self, lgb_weight=1.0, rf_weight=0.0, n_seeds=3,
        model_name="lightgbm", model_params=None,
    ):
        self.lgb_weight = lgb_weight
        self.rf_weight = rf_weight
        self.n_seeds = n_seeds
        self.model_name = model_name
        self.model_params = model_params or {}
        self.lgb_models = []
        self.rf_model = None
        self.use_ensemble = False

    def fit(self, X_train, y_train, X_valid, y_valid):
        # Lazy import avoids a module cycle while keeping training helpers in train.py.
        from .train import build_lightgbm, build_rf, build_xgboost, fit_single_model

        self.lgb_models = []
        for seed in range(42, 42 + self.n_seeds):
            if self.model_name == "xgboost":
                model = build_xgboost(seed=seed, **self.model_params)
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            else:
                model = build_lightgbm(seed=seed, **self.model_params)
                model = fit_single_model(
                    model, X_train, y_train, X_valid, y_valid, use_early_stop=True
                )
            self.lgb_models.append(model)

        if self.rf_weight > 0:
            try:
                self.rf_model = build_rf()
                self.rf_model.fit(X_train, y_train)
                self.use_ensemble = True
            except Exception as exc:
                print(f"RandomForest 训练失败：{exc}")
        return self

    def predict(self, X):
        predictions = [model.predict(X) for model in self.lgb_models]
        lgb_prediction = np.mean(predictions, axis=0)
        if not self.use_ensemble or self.rf_model is None or self.rf_weight == 0:
            return lgb_prediction
        rf_prediction = self.rf_model.predict(X)
        return self.lgb_weight * lgb_prediction + self.rf_weight * rf_prediction

    def refit_full(self, X, y):
        """Refit on all available history while preserving tuned tree counts."""
        from .train import build_lightgbm, build_rf, build_xgboost

        refitted = []
        for offset, tuned_model in enumerate(self.lgb_models):
            if self.model_name == "xgboost":
                best_iteration = int(
                    getattr(tuned_model, "best_iteration", 0) + 1
                )
                params = {
                    **self.model_params,
                    "n_estimators": best_iteration,
                    "early_stopping_rounds": None,
                }
                model = build_xgboost(seed=42 + offset, **params)
            else:
                best_iteration = int(
                    getattr(tuned_model, "best_iteration_", 0) or
                    getattr(tuned_model, "n_estimators", 10000)
                )
                model = build_lightgbm(seed=42 + offset, **self.model_params)
                model.set_params(n_estimators=best_iteration)
            model.fit(X, y)
            refitted.append(model)
        self.lgb_models = refitted

        if self.rf_weight > 0:
            self.rf_model = build_rf()
            self.rf_model.fit(X, y)
            self.use_ensemble = True
        return self
