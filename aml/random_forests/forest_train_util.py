from .RandomForestClassifier import RandomForestClassifierModel
from .RandomForestRegressor import RandomForestRegressorModel


def train_classifier_forest() -> None:
    """
    Run the random forest training for classification.
    """
    PATH = "/logs/forest_classifier.pkl"
    forest = RandomForestClassifierModel()
    forest.train_forest(cross_validation=True)
    forest.plot_learning_curve()
    forest.save_model(PATH)
    forest.evaluate_forest(PATH)


def train_regressor_forest() -> None:
    """
    Run the random forest training for regression.
    """
    PATH = "/logs/forest_regressor.pkl"
    forest = RandomForestRegressorModel()
    forest.train_forest(cross_validation=False)
    forest.save_model(PATH)
    forest.evaluate_forest(PATH)
