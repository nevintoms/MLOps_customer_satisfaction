# from zenml.steps import BaseParameters
from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """Model Configurations"""

    model_name: str = "LinearRegression"