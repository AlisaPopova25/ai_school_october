"""CatBoost as a fifth classifier for the paper's pipeline (Walch et al., SLEEP 2019).

The pipeline treats every model as a scikit-learn estimator: `ClassifierService` assigns `classifier.class_weight`,
`ParameterSearch` wraps it in `GridSearchCV` (which clones it), `multiprocessing.Pool` pickles it to workers,
and `predict_proba` must return columns in the order of the sorted labels (0 = wake, 1 = sleep / NREM, 2 = REM).

`CatBoostClassifier` itself is close to that contract, but its scikit-learn integration has changed across
versions (e.g. missing `__sklearn_tags__` with scikit-learn >= 1.6 in older CatBoost releases). This thin
wrapper is a plain scikit-learn estimator whose `fit` builds a CatBoost model, so cloning, grid search and
pickling work with any CatBoost 1.x. `catboost` is imported only inside `fit`, so the rest of the pipeline
runs (and this module imports) on machines without it.

Training (per fit):
  1. hold out the last `eval_fraction` of the rows (rows are stacked subject by subject, so the tail is mostly
     whole subjects) and train up to `iterations` trees with early stopping on it;
  2. with `refit_on_full`, retrain on all rows with the number of trees found in step 1.

Environment (all optional):
  SLEEP_CATBOOST_THREADS   CPU threads per model (default 1; the pipeline already runs SLEEP_N_JOBS processes)
  SLEEP_CATBOOST_DEVICE    CPU (default) or GPU
  SLEEP_CATBOOST_GPU_RAM   share of GPU memory per process when DEVICE=GPU (default 0.9 / SLEEP_N_JOBS)
  SLEEP_CATBOOST_GPU_DEVICES  GPU ids, default "0"
A GPU that fails to initialise falls back to CPU with a warning.

Install where you train:  pip install "catboost>=1.2"
"""
import os
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.validation import check_is_fitted

_GPU_FAILED = False  # set once per process after a failed GPU start, so clones go straight to CPU


def catboost_available():
    try:
        import catboost  # noqa: F401
        return True
    except ImportError:
        return False


class CatBoostSleepClassifier(ClassifierMixin, BaseEstimator):
    """scikit-learn compatible CatBoost classifier for sleep/wake and wake/NREM/REM.

    `class_weight` follows scikit-learn's convention ({label: weight} or 'balanced'), because
    `ClassifierService.train_classifier` sets it that way for every model.
    """

    def __init__(self, iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0, class_weight=None,
                 early_stopping_rounds=200, eval_fraction=0.15, refit_on_full=True, task_type=None,
                 random_seed=0, thread_count=None, verbose=False):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.class_weight = class_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_fraction = eval_fraction
        self.refit_on_full = refit_on_full
        self.task_type = task_type
        self.random_seed = random_seed
        self.thread_count = thread_count
        self.verbose = verbose

    # ------------------------------------------------------------------ helpers
    def _class_weights(self, y):
        if self.class_weight is None:
            return None
        if isinstance(self.class_weight, dict):
            weights = self.class_weight
        else:  # 'balanced'
            weights = dict(zip(self.classes_, compute_class_weight(self.class_weight, classes=self.classes_, y=y)))
        # CatBoost wants one weight per class, in the order of the (sorted) labels
        return [float(weights.get(c, 1.0)) for c in self.classes_]

    def _device(self):
        device = (self.task_type or os.environ.get("SLEEP_CATBOOST_DEVICE", "CPU")).upper()
        return "CPU" if device == "GPU" and _GPU_FAILED else device

    def _make(self, iterations, class_weights, device):
        from catboost import CatBoostClassifier

        threads = self.thread_count if self.thread_count is not None else int(
            os.environ.get("SLEEP_CATBOOST_THREADS", 1))
        params = dict(
            iterations=int(iterations),
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function="Logloss" if len(self.classes_) == 2 else "MultiClass",
            class_weights=class_weights,
            random_seed=self.random_seed,
            thread_count=threads,
            verbose=self.verbose,
            allow_writing_files=False,  # no catboost_info/ folders from parallel workers
            task_type=device,
        )
        if device == "GPU":
            n_jobs = max(1, int(os.environ.get("SLEEP_N_JOBS", 1)))
            params["devices"] = os.environ.get("SLEEP_CATBOOST_GPU_DEVICES", "0")
            params["gpu_ram_part"] = float(os.environ.get("SLEEP_CATBOOST_GPU_RAM", 0.9 / n_jobs))
        return CatBoostClassifier(**params)

    def _fit_on_device(self, iterations, X, y, sample_weight, class_weights, eval_set=None):
        device = self._device()
        kwargs = dict(sample_weight=sample_weight)
        if eval_set is not None:
            kwargs.update(eval_set=eval_set, early_stopping_rounds=self.early_stopping_rounds, use_best_model=True)
        try:
            model = self._make(iterations, class_weights, device).fit(X, y, **kwargs)
        except Exception as error:  # CatBoostError: GPU not available / unsupported architecture
            if device != "GPU":
                raise
            global _GPU_FAILED
            _GPU_FAILED = True
            warnings.warn(f"CatBoost GPU training failed ({error}); falling back to CPU for this process")
            model = self._make(iterations, class_weights, "CPU").fit(X, y, **kwargs)
        return model

    def _holdout(self, y):
        """Tail of the rows (≈ whole subjects) as early-stopping set; random rows if the tail misses a class."""
        n = len(y)
        n_eval = int(round(n * self.eval_fraction))
        if not self.early_stopping_rounds or n_eval < 50 or n - n_eval < 50:
            return None
        eval_idx = np.arange(n - n_eval, n)
        if len(np.unique(y[eval_idx])) < len(self.classes_) or len(np.unique(y[: n - n_eval])) < len(self.classes_):
            eval_idx = np.sort(np.random.default_rng(self.random_seed).choice(n, n_eval, replace=False))
        train_mask = np.ones(n, dtype=bool)
        train_mask[eval_idx] = False
        return train_mask

    # ------------------------------------------------------------------ sklearn API
    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("CatBoostSleepClassifier needs at least two classes in the training labels")
        self.n_features_in_ = X.shape[1]
        class_weights = self._class_weights(y)
        # labels are passed as 0..K-1 so CatBoost's class order is guaranteed to match self.classes_
        y_index = np.searchsorted(self.classes_, y)
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

        train_mask = self._holdout(y_index)
        if train_mask is None:
            self.best_iteration_ = int(self.iterations)
            self.model_ = self._fit_on_device(self.iterations, X, y_index, sw, class_weights)
            return self

        model = self._fit_on_device(self.iterations, X[train_mask], y_index[train_mask],
                                    None if sw is None else sw[train_mask], class_weights,
                                    eval_set=(X[~train_mask], y_index[~train_mask]))
        best = model.get_best_iteration()
        self.best_iteration_ = int(self.iterations if best is None else best + 1)
        if self.refit_on_full:
            # retrain on every row with the tree count found by early stopping (+15% for the extra data)
            trees = max(50, int(round(self.best_iteration_ * (1 + self.eval_fraction))))
            self.model_ = self._fit_on_device(trees, X, y_index, sw, class_weights)
        else:
            self.model_ = model
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        proba = np.asarray(self.model_.predict_proba(np.asarray(X, dtype=float)))
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def feature_importances_(self):
        check_is_fitted(self, "model_")
        return np.asarray(self.model_.get_feature_importance())
