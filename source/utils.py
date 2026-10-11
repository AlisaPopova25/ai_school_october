from io import StringIO
from pathlib import Path

import numpy as np
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from source.analysis.setup.attributed_classifier import AttributedClassifier
from source.analysis.setup.feature_type import FeatureType


class LiblinearLogisticRegression(LogisticRegression):
    """LogisticRegression(solver='liblinear') with the multiclass behaviour of scikit-learn 0.20 (used in the paper).

    Old sklearn silently fitted liblinear one-vs-rest on 3 classes; sklearn >= 1.8 raises an error instead.
    For 3 classes we fit one binary model per class (class weights applied as sample weights by the original
    label, like sklearn 0.20 did) and normalise the per-class sigmoids, exactly as multi_class='ovr' did.
    """

    def fit(self, X, y, sample_weight=None):
        classes = np.unique(y)
        if len(classes) <= 2:
            return super().fit(X, y, sample_weight)
        from sklearn.utils.class_weight import compute_class_weight
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if self.class_weight is not None:
            cw = self.class_weight if isinstance(self.class_weight, dict) else dict(
                zip(classes, compute_class_weight(self.class_weight, classes=classes, y=y)))
            weights = weights * np.array([cw[c] for c in y])
        params = self.get_params()
        params['class_weight'] = None
        binary = [LogisticRegression(**params).fit(X, (y == c).astype(int), sample_weight=weights) for c in classes]
        self.classes_ = classes
        self.coef_ = np.vstack([m.coef_ for m in binary])
        self.intercept_ = np.concatenate([m.intercept_ for m in binary])
        self.n_iter_ = np.concatenate([np.atleast_1d(m.n_iter_) for m in binary])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        if len(self.classes_) <= 2:
            return super().predict_proba(X)
        p = 1 / (1 + np.exp(-self.decision_function(X)))
        return p / p.sum(axis=1, keepdims=True)


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_classifiers():
    return [AttributedClassifier(name='Random Forest',
                                 classifier=RandomForestClassifier(n_estimators=100, max_features=1.0,
                                                                   max_depth=10,
                                                                   min_samples_split=10, min_samples_leaf=32,
                                                                   bootstrap=True)),
            AttributedClassifier(name='Logistic Regression',
                                 classifier=LiblinearLogisticRegression(penalty='l1', solver='liblinear', verbose=0)),
            AttributedClassifier(name='k-Nearest Neighbors',
                                 classifier=KNeighborsClassifier(weights='distance')),
            AttributedClassifier(name='Neural Net',
                                 classifier=MLPClassifier(activation='relu', hidden_layer_sizes=(15, 15, 15),
                                                          max_iter=2000, alpha=0.01, solver='adam', verbose=False,
                                                          n_iter_no_change=20))] + get_catboost_classifiers()


def get_catboost_classifiers():
    """CatBoost joins every classifier list when `catboost` is installed (SLEEP_INCLUDE_CATBOOST=0 turns it off)."""
    import os
    from source.analysis.classification.catboost_classifier import CatBoostSleepClassifier, catboost_available
    if os.environ.get('SLEEP_INCLUDE_CATBOOST', '1') == '0':
        return []
    if not catboost_available():
        print('CatBoost is skipped: install it with  pip install "catboost>=1.2"')
        return []
    from source.analysis.classification import catboost_config  # all CatBoost settings live there
    return [AttributedClassifier(name='CatBoost', classifier=CatBoostSleepClassifier(**catboost_config.PARAMS))]


def get_base_feature_sets():
    from source.constants import Constants
    if Constants.INCLUDE_CIRCADIAN:  # the paper's main results use the circadian-model clock proxy
        return [[FeatureType.count],
                [FeatureType.heart_rate],
                [FeatureType.count, FeatureType.heart_rate],
                [FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model]]
    return [[FeatureType.count],
            [FeatureType.heart_rate],
            [FeatureType.count, FeatureType.heart_rate],
            [FeatureType.count, FeatureType.heart_rate, FeatureType.cosine]]
    # 12-23-19 note: I'm making the default base feature use cosine, not circadian model
    # so that it doesn't require MATLAB to run


def convert_pdf_to_txt(pdf_path_string, all_texts):
    resource_manager = PDFResourceManager()
    returned_string = StringIO()
    codec = 'utf-8'
    layout_parameters = LAParams(all_texts=all_texts)
    device = TextConverter(resource_manager, returned_string, codec=codec, laparams=layout_parameters)
    fp = open(pdf_path_string, 'rb')
    interpreter = PDFPageInterpreter(resource_manager, device)
    password = ""
    max_pages = 0
    caching = True
    page_numbers = set()

    for page in PDFPage.get_pages(fp, page_numbers, maxpages=max_pages, password=password, caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = returned_string.getvalue()

    fp.close()
    device.close()
    returned_string.close()
    return text


def smooth_gauss(y, box_pts):
    box = np.ones(box_pts) / box_pts
    mu = int(box_pts / 2.0)
    sigma = 50  # seconds

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu) / sigma) ** 2))

    box = box / np.sum(box)
    sum_value = 0
    for ind in range(0, box_pts):
        sum_value += box[ind] * y[ind]

    return sum_value


def convolve_with_dog(y, box_pts):
    y = y - np.mean(y)
    box = np.ones(box_pts) / box_pts

    mu1 = int(box_pts / 2.0)
    sigma1 = 120

    mu2 = int(box_pts / 2.0)
    sigma2 = 600

    scalar = 0.75

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu1) / sigma1) ** 2)) - scalar * np.exp(
            -1 / 2 * (((ind - mu2) / sigma2) ** 2))

    y = np.insert(y, 0, np.flip(y[0:int(box_pts / 2)]))  # Pad by repeating boundary conditions
    y = np.insert(y, len(y) - 1, np.flip(y[int(-box_pts / 2):]))
    y_smooth = np.convolve(y, box, mode='valid')

    return y_smooth


def remove_repeats(array):
    array_no_repeats = np.unique(array, axis=0)
    array_no_repeats = array_no_repeats[np.argsort(array_no_repeats[:, 0])]
    return array_no_repeats


def remove_nans(array):
    array = array[~np.isnan(array).any(axis=1)]
    array = array[~np.isinf(array).any(axis=1)]
    return array
