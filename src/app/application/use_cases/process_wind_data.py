from datetime import date
import logging
from typing import List
from domain.interfaces import WindVelocityDataSource, WindVelocityRepository, WindVelocityAvgRepository
from domain.entities import WindVelocity
from domain.services import WindVelocityService


def process_wind_velocity_data(
    data_source: WindVelocityDataSource,
    start_date: date,
    end_date: date,
    region: str,
    data_target: WindVelocityRepository,
    avg_data_target: (WindVelocityRepository, None) = None,
    velocity_service: (WindVelocityService, None) = None,
) -> List[WindVelocity]:
    wind_velocities = data_source.fetch(start_date, end_date, region)
    logging.info(f'Got {len(wind_velocities)} wind velocities from {type(data_source).__name__}')

    data_target.save(wind_velocities)
    logging.info(f'Saved {len(wind_velocities)} wind velocities to {type(data_target).__name__}')

    # Now get the averages and save to provided target for averages:
    if avg_data_target and velocity_service:
        averages = velocity_service.calculate_daily_avg(wind_velocities)
        avg_data_target.save(averages)
        logging.info(f'Saved {len(averages)} wind velocity daily averages to {type(avg_data_target).__name__}')

    return wind_velocities
