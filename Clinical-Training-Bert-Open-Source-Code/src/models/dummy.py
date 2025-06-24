from typing import Union, Optional

import pandas as pd
from numpy.typing import NDArray
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler


class ReferenceClassifier:
    def __init__(self, scaler: Optional[StandardScaler] = None):
        """
        Args:
            scaler (Optional[StandardScaler]): A standard scaler used to normalize the inputs before training.
        """
        self.scaler = scaler
        self.model = None

    def fit(
        self,
        X: Union[NDArray, pd.DataFrame],
        y: Union[NDArray, pd.DataFrame],
    ) -> None:
        """
        Trains reference model.

        Args:
            X (Union[NDArray, pd.DataFrame]): Input predictors.
            y (Union[NDArray, pd.DataFrame]): Input labels.
        """
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = X_scaled
            
        if self.model is not None:
            self.model.fit(X, y)

    def predict(self, X: NDArray) -> Union[NDArray, None]:
        """
        Makes prediction using trained reference model.

        Args:
            X (Union[NDArray, pd.DataFrame]): Input predictors.
            y (Union[NDArray, pd.DataFrame]): Input labels.

        Returns:
            NDArray: Predicted probabilities.
        """
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = X_scaled
        
        if self.model is not None:
            preds = self.model.predict_proba(X)
            return preds[:, 1]


class RandomClassifier(ReferenceClassifier):
    """Reference classifier that makes random stratified T1D predictions."""    
    def __init__(self, scaler: Optional[StandardScaler] = None):

        """
        Args:
            scaler (Optional[StandardScaler]): A standard scaler used to normalize the inputs before training.
        """
        super().__init__(scaler)
        self.model = DummyClassifier(strategy="stratified")


class ConstantClassifier(ReferenceClassifier):
    """Reference classifier that makes constant T1D predictions."""    
    def __init__(self, scaler: Optional[StandardScaler] = None):

        """
        Args:
            scaler (Optional[StandardScaler]): A standard scaler used to normalize the inputs before training.
        """
        super().__init__(scaler)
        self.model = DummyClassifier(strategy="constant", constant=1)
