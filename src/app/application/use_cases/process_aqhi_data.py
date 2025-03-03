from datetime import date
from typing import List
import logging

from domain.interfaces import AQHIDataSource, AQHIRepository
from domain.entities import AQHI


def process_aqhi_data(
    data_source: AQHIDataSource,
    start_date: date,
    end_date: date,
    region: str,
    data_target: AQHIRepository
) -> List[AQHI]:
    aqhi_data = data_source.fetch(start_date, end_date, region)
    logging.info(f'Got {len(aqhi_data)} AQHI data points from {type(data_source).__name__}')

    data_target.save(aqhi_data)
    logging.info(f'Saved {len(aqhi_data)} AQHI data points to {type(data_target).__name__}')

    return aqhi_data
