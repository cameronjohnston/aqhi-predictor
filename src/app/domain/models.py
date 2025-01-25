from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
import math
from typing import List, Optional


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


@dataclass
class WindVelocityAvg:
    latitude: float
    longitude: float
    observed_date: date
    speed: float
    source_direction: int  # In degrees, with 0 = North and increasing going clockwise (meteorological convention)
    source: str
    forecast_datetime: (datetime, None) = None


class WindVelocityService:
    def calculate_daily_avg(self, wind_velocities: List[WindVelocity]) -> List[WindVelocityAvg]:
        # Group wind velocities by latitude, longitude, and observed date
        grouped_data = defaultdict(list)

        for velocity in wind_velocities:
            observed_date = velocity.observed_datetime.date()
            grouped_data[(velocity.latitude, velocity.longitude, observed_date, velocity.source)].append(velocity)

        # Now we have the grouped_data containing a list for each grouping
        daily_averages = []
        for (latitude, longitude, observed_date, source), velocities in grouped_data.items():
            # Initialize components
            total_x = total_y = 0.0
            count = len(velocities)

            for v in velocities:
                # Convert direction to radians ccw from westerly
                direction_rad = self.meteorological2math(v.source_direction)

                # Add vector components (eastward, northward)
                total_x += v.speed * math.sin(direction_rad)
                total_y += v.speed * math.cos(direction_rad)

            # Calculate the resultant vector
            avg_x = total_x / count
            avg_y = total_y / count
            avg_speed = math.sqrt(avg_x ** 2 + avg_y ** 2)

            # Calculate the average direction (in radians)
            avg_direction_math = math.atan2(avg_x, avg_y)

            # Convert the angle to the meteorological convention, in degrees
            avg_direction_met = self.math2meteorological(avg_direction_math)

            daily_averages.append(WindVelocityAvg(
                latitude=latitude,
                longitude=longitude,
                observed_date=observed_date,
                speed=avg_speed,
                source_direction=int(round(avg_direction_met)),  # Rounded to nearest integer
                source=source,
            ))

        return daily_averages

    def meteorological2math(self, degrees_cw_from_northerly: float) -> float:
        """
        Convert from meteorological convention (degrees clockwise from North)
        to mathematical convention (radians counterclockwise from West).
        """
        return math.radians((270 - degrees_cw_from_northerly) % 360)

    def math2meteorological(self, radians_ccw_from_westerly: float) -> float:
        """
        Convert from mathematical convention (radians counterclockwise from West)
        to meteorological convention (degrees clockwise from North).
        """
        return (270 - math.degrees(radians_ccw_from_westerly)) % 360

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


@dataclass
class AQHI:
    """ Air Quality Health Index """
    latitude: float
    longitude: float
    observed_datetime: datetime
    value: float
    source: str
    forecast_datetime: (datetime, None) = None


