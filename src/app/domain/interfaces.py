from abc import ABC, abstractmethod
from datetime import date
from typing import List
from domain.models import Wildfire, WindVelocity, WindVelocityAvg, AQHI


class WildfireDataSource(ABC):
    @abstractmethod
    def fetch(self, start_date: date, end_date: date, region: str) -> List[Wildfire]:
        pass


class WildfireRepository(ABC):
    @abstractmethod
    def save(self, wildfires: List[Wildfire]) -> None:
        pass


class WindVelocityDataSource(ABC):
    @abstractmethod
    def fetch(self, start_date: date, end_date: date, region: str) -> List[WindVelocity]:
        pass


class WindVelocityRepository(ABC):
    @abstractmethod
    def save(self, velocities: List[WindVelocity]) -> None:
        pass


class WindVelocityAvgRepository(ABC):
    @abstractmethod
    def save(self, velocities: List[WindVelocityAvg]) -> None:
        pass


class AQHIDataSource(ABC):
    @abstractmethod
    def fetch(self, start_date: date, end_date: date, region: str) -> List[AQHI]:
        pass


class AQHIRepository(ABC):
    @abstractmethod
    def save(self, velocities: List[AQHI]) -> None:
        pass



