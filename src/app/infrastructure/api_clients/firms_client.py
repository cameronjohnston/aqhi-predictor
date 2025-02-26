import csv
from io import StringIO
import requests
from typing import List
from datetime import date, timedelta
from domain.interfaces import WildfireDataSource
from domain.entities import Wildfire
from infrastructure.config import load_config


class FIRMSClient(WildfireDataSource):
    region_to_box = {
        # Format of a box is: west, south, east, north
        # TODO: expand on below.
        'VANCOUVER': '-133,39,-113,59'
    }
    def __init__(self):
        config = load_config()
        self.base_url = config["firms_api"]["base_url"]
        self.api_key = config["firms_api"]["api_key"]
        self.format = config["firms_api"]["format"]
        self.source = config["firms_api"]["source"]

    def fetch(self, start_date: date, end_date: date, region: str = 'VANCOUVER') -> List[Wildfire]:
        """Fetch wildfire data in chunks of max 10 days at a time (FIRMS API restriction)."""
        box = self.region_to_box.get(region)
        all_wildfires = []

        current_start = start_date
        while current_start <= end_date:
            current_end = min(current_start + timedelta(days=9), end_date)  # Max 10 days at a time

            num_days = (current_end - current_start + timedelta(days=1)).days
            url = f"{self.base_url}/api/area/{self.format}/{self.api_key}/{self.source}/{box}/{num_days}/{current_end.isoformat()}"

            response = requests.get(url)
            response.raise_for_status()

            # Parse CSV response
            csv_reader = csv.DictReader(StringIO(response.text))
            data = [row for row in csv_reader]

            # Convert to Wildfire objects
            wildfires = [
                Wildfire(
                    id=f"{item['latitude']}-{item['longitude']}-{item['acq_date']}",
                    first_discovered=date.fromisoformat(item["acq_date"]),  # TODO: refine matching logic
                    latitude=float(item["latitude"]),
                    longitude=float(item["longitude"]),
                    frp=float(item["frp"]),
                    asofdate=date.fromisoformat(item["acq_date"]),
                    source="FIRMS",
                )
                for item in data
            ]

            all_wildfires.extend(wildfires)
            current_start = current_end + timedelta(days=1)  # Move to next batch

        return all_wildfires

    def fetch_OLD(self, start_date: date, end_date: date, region: str = 'VANCOUVER') -> List[Wildfire]:
        box = self.region_to_box.get(region)
        num_days = (end_date - start_date + timedelta(days=1)).days
        url = f"{self.base_url}/api/area/{self.format}/{self.api_key}/{self.source}/{box}/{num_days}/{end_date.isoformat()}"

        response = requests.get(url)
        response.raise_for_status()

        # Response text will be a string. Use StringIO to treat the string as a file object
        csv_reader = csv.DictReader(StringIO(response.text))
        data = [row for row in csv_reader]

        # TODO_DQ: check for non-null and "reasonable" values first?
        return [
            Wildfire(
                id=f"{item['latitude']}-{item['longitude']}",
                first_discovered=date.fromisoformat(item["acq_date"]),  # TODO: some matching to accurately populate
                latitude=item["latitude"],
                longitude=item["longitude"],
                frp=item["frp"],
                asofdate=date.fromisoformat(item["acq_date"]),
                source="FIRMS",
            )
            for item in data
        ]
