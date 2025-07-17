import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, RMSE, R2
from typing_extensions import Annotated
from typing import Tuple
from sklearn.base import RegressorMixin

@step
def evaluate_model(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,           
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    """
    Evaluates the model on the ingested data.

    Args:
        df: the ingested data
    """
    try:
        prediction = model.predict(X_test)

        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)

        r2_class = R2()
        r2_score = r2_class.calculate_score(y_test, prediction)

        # Using the RMSE class for root mean squared error calculation
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)

        return r2_score, rmse
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e