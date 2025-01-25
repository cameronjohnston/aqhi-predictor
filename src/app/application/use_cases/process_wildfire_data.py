from datetime import date
from typing import List
from domain.interfaces import WildfireDataSource, WildfireRepository
from domain.models import Wildfire


def process_wildfire_data(data_source: WildfireDataSource, start_date: date, end_date: date, region: str, data_target: WildfireRepository) -> List[Wildfire]:
    wildfires = data_source.fetch(start_date, end_date, region)
    print(f'Got {len(wildfires)} wildfires from {type(data_source).__name__}')

    data_target.save(wildfires)
    return wildfires
