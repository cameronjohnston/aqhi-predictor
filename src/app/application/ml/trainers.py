
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from domain.entities import AQHI, Wildfire, WindVelocity


@dataclass
class ModelTrainer(ABC):
    """ Interface for model trainer(s) """
    wildfires: List[Wildfire]
    wind_velocities: List[WindVelocity]
    aqhi_data: List[AQHI]
    time_intervals: List[int]  # List of time intervals in hours (e.g., [6, 12, 24, 48])

    @abstractmethod
    def train(self) -> None:
        """ Train the model """
        pass

    @abstractmethod
    def prepare_training_data(self):
        """ Prepare training data """
        pass

    @abstractmethod
    def get_training_data(self):
        """ Return processed training data in a format agnostic to ML frameworks """
        pass

    @property
    def cn(self):
        return type(self).__name__

