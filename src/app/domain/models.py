from dataclasses import dataclass
from datetime import date, datetime
import math
from typing import List, Optional

from domain.util.wind import x_y_components  # WindVelocityService


@dataclass
class Wildfire:
    id: str
    first_discovered: date
    latitude: float
    longitude: float
    frp: float  # Fire Radiative Power
    asofdate: date
    source: str


@dataclass
class WindVelocity:
    latitude: float
    longitude: float
    observed_datetime: datetime
    speed: float
    source_direction: int  # In degrees, with 0 = North and increasing going clockwise (meteorological convention)
    source: str
    forecast_datetime: (datetime, None) = None

    def __post_init__(self):
        """ Compute and assign the X and Y components of velocity """
        self.add_x_y_components()

    def add_x_y_components(self):
        """ Provide X and Y components of velocity as a tuple (x, y) """
        self.x_component, self.y_component = x_y_components(self)


@dataclass
class WindVelocityAvg:
    latitude: float
    longitude: float
    observed_date: date
    speed: float
    source_direction: int  # In degrees, with 0 = North and increasing going clockwise (meteorological convention)
    source: str
    forecast_datetime: (datetime, None) = None

    def __post_init__(self):
        """ Compute and assign the X and Y components of velocity """
        self.add_x_y_components()

    def add_x_y_components(self):
        """ Provide X and Y components of velocity as a tuple (x, y) """
        self.x_component, self.y_component = x_y_components(self)


@dataclass
class AQHI:
    """ Air Quality Health Index """
    latitude: float
    longitude: float
    observed_datetime: datetime
    value: float
    source: str
    forecast_datetime: (datetime, None) = None


# TODO_CLEANUP: retire if not used?
@dataclass
class StationMetadata:
    station_id: int
    station_name: str
    latitude: float
    longitude: float
    elevation: Optional[float]
    source: str

@dataclass
class BBox:
    west: float
    south: float
    east: float
    north: float

    @classmethod
    def from_string(cls, bbox_str: str) -> "BBox":
        """Parses a comma-delimited bounding box string into a BBox instance."""
        west, south, east, north = map(float, bbox_str.split(","))
        return cls(west, south, east, north)

    def contains(self, latitude: float, longitude: float) -> bool:
        """Checks if a given coordinate is inside the bounding box."""
        return self.west <= longitude <= self.east and self.south <= latitude <= self.north



