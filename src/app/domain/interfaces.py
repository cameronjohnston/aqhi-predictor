from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional
from domain.models import Wildfire, WindVelocity, WindVelocityAvg, AQHI, BBox


class WildfireDataSource(ABC):
    @abstractmethod
    def fetch(self, start_date: date, end_date: date, region: str) -> List[Wildfire]:
        pass


class WildfireRepository(ABC):
    @abstractmethod
    def save(self, wildfires: List[Wildfire]) -> None:
        pass

    @abstractmethod
    def get(self, bbox: Optional[BBox] = None, start_date: Optional[date] = None, end_date: Optional[date] = None
            ) -> List[Wildfire]:
        pass


class WindVelocityDataSource(ABC):
    @abstractmethod
    def fetch(self, start_date: date, end_date: date, region: str) -> List[WindVelocity]:
        pass


class WindVelocityRepository(ABC):
    @abstractmethod
    def save(self, velocities: List[WindVelocity]) -> None:
        pass

    @abstractmethod
    def get(self, bbox: Optional[BBox] = None, start_date: Optional[date] = None, end_date: Optional[date] = None,
            is_forecast: bool = False) -> List[WindVelocity]:
        pass


class WindVelocityAvgRepository(ABC):
    @abstractmethod
    def save(self, velocities: List[WindVelocityAvg]) -> None:
        pass

    @abstractmethod
    def get(self, bbox: Optional[BBox] = None, start_date: Optional[date] = None, end_date: Optional[date] = None,
            is_forecast: bool = False) -> List[WindVelocityAvg]:
        pass


class AQHIDataSource(ABC):
    @abstractmethod
    def fetch(self, start_date: date, end_date: date, region: str) -> List[AQHI]:
        pass


class AQHIRepository(ABC):
    @abstractmethod
    def save(self, velocities: List[AQHI]) -> None:
        pass

    @abstractmethod
    def get(self, bbox: Optional[BBox] = None, start_date: Optional[date] = None, end_date: Optional[date] = None, 
            is_forecast: bool = False) -> List[AQHI]:
        pass


