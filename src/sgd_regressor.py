from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV


class OptimizedSGDRegressor:
    def __init__(self):
        self.best_model = None
        self.best_configuration = None

    def fit(self, X, y):
        # Define the parameter grid for the grid search
        param_grid = {
            'loss': ['squared_loss'],  # Loss functions
            'penalty': ['elasticnet'],  # Regularization penalties
            'alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],  # Regularization strength
            'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],  # ElasticNet mixing parameter
            'max_iter': [10, 100, 1000],  # Maximum number of iterations
            'learning_rate': ['constant', 'invscaling', 'adaptive'],  # Learning rate schedule
            'eta0': [0.001, 0.01, 0.1]  # Initial learning rate
        }

        # Create the SGDRegressor model
        sgd_regressor = SGDRegressor()

        # Create the GridSearchCV object
        grid_search = GridSearchCV(sgd_regressor, param_grid, cv=int(len(X)/2), n_jobs=-1, verbose=2)

        # Fit the GridSearchCV to find the best hyperparameterization
        grid_search.fit(X, y)

        # Get the best model instance
        self.best_model = grid_search.best_estimator_
        self.best_configuration = grid_search.best_params_

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("Model has not been trained. Please call fit() before using predict().")
        return self.best_model.predict(X)

    def get_best_configuration(self):
        if self.best_configuration is None:
            raise ValueError("Model has not been trained. Please call fit() before retrieving the best configuration.")
        return self.best_configuration
