
from abc import ABC, abstractmethod


class AQHIPredictor(ABC):
    """ Interface for AQHI prediction models """

    @abstractmethod
    def predict(self):
        """ Predict AQHI based on input features """
        pass

    @property
    def cn(self):
        return type(self).__name__


