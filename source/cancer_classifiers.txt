"""
This file contains useful functions for creating cancer classification models.

Alex Angus
Gray Selby

October 20, 2019
"""

import numpy as np
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
        
def random_hyperparameter_search(trained_model, hyperparameters, iterations,
                                 cross_validation_folds, X_train, y_train,
                                 X_test, y_test):
    print("Performing Random Hyperparameter Search with hyperparameters:")
    pprint(hyperparameters)
    random_search = RandomizedSearchCV(estimator=trained_model,
                                          param_distributions=hyperparameters,
                                          n_iter=iterations,
                                          cv=cross_validation_folds)
    random_search.fit(X_train, y_train)
    print("Best hyperparameters: ", random_search.best_params_)
    print("Best hyperparameter accuracy with {} iterations: ".format(iterations), random_search.score(X_test, y_test))

def grid_hyperparameter_search(trained_model, hyperparameters,
                               cross_validation_folds, X_train, y_train,
                               X_test, y_test):
    print("Performing Hyperparameter Grid Search with hyperparameters:")
    pprint(hyperparameters)
    grid_search = GridSearchCV(estimator=trained_model,
                               param_grid=hyperparameters,
                               cv=cross_validation_folds,
                               n_jobs=-1)
                               
    grid_search.fit(X_train, y_train)
    print("Best hyperparameters: ", grid_search.best_params_)
    print("Best hyperparameter accuracy: ", grid_search.score(X_test, y_test))
    