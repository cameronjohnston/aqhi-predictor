
from collections import defaultdict
import math
from typing import Dict, List, Tuple

from domain.util.wind import meteorological2math, math2meteorological


class WindVelocityService:
    from domain.models import WindVelocityAvg

    @staticmethod
    def calculate_daily_avg(wind_velocities: List["WindVelocity"]) -> List[WindVelocityAvg]:
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
                direction_rad = meteorological2math(v.source_direction)

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
            avg_direction_met = math2meteorological(avg_direction_math)

            daily_averages.append(WindVelocityAvg(
                latitude=latitude,
                longitude=longitude,
                observed_date=observed_date,
                speed=avg_speed,
                source_direction=int(round(avg_direction_met)),  # Rounded to nearest integer
                source=source,
            ))

        return daily_averages


class WildfireImpactScorer:
    """Calculates the estimated impact of wildfires on AQHI."""
    from domain.models import Wildfire, WindVelocity

    def __init__(self, aqhi_locations: List[Tuple[float, float]], hrs_ahead: List[int], decay_factor: float = 0.5):
        """
        :param aqhi_locations: List of coordinates for which to score AQHI impact (in pairs of latitude, longitude)
        :param hrs_ahead: List of how many hours ahead to produce scores for (e.g. 24h, 48h, ...)
        :param decay_factor: How much the impact of a wildfire decreases over time.
        """
        self.aqhi_locations = aqhi_locations
        self.hrs_ahead = hrs_ahead
        self.decay_factor = decay_factor

    def score_wildfire(self, wildfire: Wildfire, wind_velocities: List[WindVelocity]
                       ) -> Dict[Tuple[float, float], Dict[int, float]]:
        pass  # TODO: implement

    def _compute_wind_direction_factor(self, wind_dir: int, wildfire: Wildfire,
                                       aqhi_location: Tuple[float, float]) -> float:
        """
        Computes how much the wind is blowing towards the AQHI location.
        Returns a factor from 0 (not towards AQHI) to 1 (directly towards AQHI).
        """
        from math import atan2, degrees, cos, radians

        fire_lat, fire_lon = wildfire.latitude, wildfire.longitude
        aqhi_lat, aqhi_lon = aqhi_location

        # Compute direction from wildfire to AQHI location
        delta_lat = aqhi_lat - fire_lat
        delta_lon = aqhi_lon - fire_lon
        angle_to_aqhi = (degrees(atan2(delta_lon, delta_lat)) + 360) % 360  # Normalize 0-360°

        # Wind direction is where the wind is coming FROM (0° = north wind, blows south)
        # Compute cosine similarity between wind direction and fire-to-AQHI direction
        wind_dir_rad = radians(wind_dir)
        aqhi_dir_rad = radians(angle_to_aqhi)

        return max(0, cos(aqhi_dir_rad - wind_dir_rad))  # 1 if aligned, 0 if opposite

