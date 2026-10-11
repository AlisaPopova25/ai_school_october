from sklearn.model_selection import GridSearchCV

from source.analysis.classification import catboost_config


class ParameterSearch(object):
    parameter_dictionary = {
        'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']},
        'Random Forest': {'max_depth': [10, 50, 100]},
        'k-Nearest Neighbors': {'n_neighbors': [500, 1000]},
        'Neural Net': {'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001]},
        'CatBoost': catboost_config.GRID  # edit in catboost_config.py
    }

    @staticmethod
    def run_search(attributed_classifier, training_x, training_y, scoring):
        parameter_range = ParameterSearch.parameter_dictionary[attributed_classifier.name]
        if all(len(values) == 1 for values in parameter_range.values()):
            # a single combination: nothing to search, skip the 3-fold cross-validation
            return {name: values[0] for name, values in parameter_range.items()}
        grid_search = GridSearchCV(attributed_classifier.classifier, parameter_range, scoring=scoring, cv=3)
        grid_search.fit(training_x, training_y)
        return grid_search.best_params_
