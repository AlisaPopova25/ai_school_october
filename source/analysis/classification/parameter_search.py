from sklearn.model_selection import GridSearchCV


class ParameterSearch(object):
    parameter_dictionary = {
        'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']},
        'Random Forest': {'max_depth': [10, 50, 100]},
        'k-Nearest Neighbors': {'n_neighbors': [500, 1000]},
        'Neural Net': {'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001]},
        'CatBoost': {'depth': [4, 6, 8], 'learning_rate': [0.03, 0.1], 'l2_leaf_reg': [3, 10]}
    }

    @staticmethod
    def run_search(attributed_classifier, training_x, training_y, scoring):
        parameter_range = ParameterSearch.parameter_dictionary[attributed_classifier.name]
        grid_search = GridSearchCV(attributed_classifier.classifier, parameter_range, scoring=scoring, cv=3)
        grid_search.fit(training_x, training_y)
        return grid_search.best_params_
