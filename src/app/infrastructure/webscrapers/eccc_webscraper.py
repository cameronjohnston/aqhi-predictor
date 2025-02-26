from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta, timezone
import json
import logging
import pandas as pd
import re
import requests
from typing import Dict, List

from domain.interfaces import WindVelocityDataSource, AQHIDataSource
from domain.entities import AQHI, BBox, WindVelocity, WindVelocityAvg
from infrastructure.config import load_config


class ECCCWindVelocityForecastWebscraper(WindVelocityDataSource):
    region_to_bbox = {
        # Format of a box is: west, south, east, north
        # TODO: expand on below.
        'VANCOUVER': '-123.5,48.5,-122.5,49.5'
    }

    def __init__(self):
        config = load_config()
        self.base_url = config["eccc_webscraping_wind_velocity_forecast"]["base_url"]
        self.vertical_level_type = config["eccc_webscraping_wind_velocity_forecast"]["vertical_level_type"]

        # Expecting the below to be a comma-delimited list of ints padded with leading zero(s) to be two digits
        model_run_starts_included_str = config["eccc_webscraping_wind_velocity_forecast"]["model_run_starts_included"]
        self.model_run_starts_included = model_run_starts_included_str.split(',')

        # Expecting the below to be a comma-delimited list of ints padded with leading zero(s) to be three digits
        hours_ahead_included_str = config["eccc_webscraping_wind_velocity_forecast"]["hours_ahead_included"]
        self.hours_ahead_included = hours_ahead_included_str.split(',')

    def base_dir(self, model_run_start: str, hours_ahead: str) -> str:
        return f"{self.base_url}/{model_run_start}/{hours_ahead}"

    def list_files(self, model_run_start: str, hours_ahead: str) -> List[str]:
        """ Find files located at the directory corresponding to the requested model run start & hours ahead """
        dir_url = self.base_dir(model_run_start, hours_ahead)
        response = requests.get(dir_url)
        response.raise_for_status()  # Ensure we notice bad responses

        soup = BeautifulSoup(response.text, 'html.parser')
        files = []

        # Find all 'a' tags with 'href' attribute
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Skip parent directory link
            if href not in ('../', '/'):
                files.append(href)

        return files

    def matching_files(self, model_run_start: str, hours_ahead: str, variable: str) -> List[str]:
        """ Find relevant file(s) from MSC Datamart - though only one is expected """
        file_pattern = re.compile(
            r"(\d{8})T" + re.escape(model_run_start) + r"Z_MSC_RDPS-UMOS-MLR_" + re.escape(variable) + "_"
            + re.escape(self.vertical_level_type) + r"_PT" + re.escape(hours_ahead) + r"H\.json$"
        )
        all_files = self.list_files(model_run_start, hours_ahead)
        return [f for f in all_files if file_pattern.match(f)]

    def match_wind_forecasts(self, wind_speeds: List[Dict], wind_directions: List[Dict], source: str
                             ) -> List[WindVelocity]:
        """Matches wind speed and direction forecasts based on their common ID structure."""

        # Convert lists into dictionaries for fast lookup, using a normalized key
        def extract_common_id(forecast):
            return forecast["id"].replace("WindSpeed", "").replace("WindDir", "")

        speed_dict = {extract_common_id(ws): ws for ws in wind_speeds}
        direction_dict = {extract_common_id(wd): wd for wd in wind_directions}

        velocities = []

        for key in speed_dict.keys() & direction_dict.keys():  # Intersection of keys
            speed = speed_dict[key]
            direction = direction_dict[key]

            # Extract latitude & longitude
            latitude, longitude, _ = speed["geometry"]["coordinates"]

            # Create & append WindVelocity instance
            velocities.append(WindVelocity(
                latitude=speed['geometry']['coordinates'][1],
                longitude=speed['geometry']['coordinates'][0],
                observed_datetime=datetime.fromisoformat(speed['properties']['reference_datetime']),
                speed=speed['properties']['forecast_value'],
                source_direction=direction['properties']['forecast_value'],  # Degrees clockwise of North
                source=source,
                forecast_datetime=datetime.fromisoformat(speed['properties']['forecast_datetime']),
            ))

        logging.info(f"Matched {len(velocities)} wind forecast pairs.")
        return velocities

    def fetch(self, start_date: date, end_date: date, region: str = 'VANCOUVER') -> List[WindVelocity]:
        bbox_str = self.region_to_bbox.get(region)
        bbox = BBox.from_string(bbox_str)
        all_velocities = []

        # Loop through model run starts and hours ahead which are included
        for mrs in self.model_run_starts_included:
            for ha in self.hours_ahead_included:
                wind_dir_files = self.matching_files(mrs, ha, 'WindDir')
                wind_speed_files = self.matching_files(mrs, ha, 'WindSpeed')

                # TODO: Sanity check? There should be exactly 1 of each
                wind_dir_file = wind_dir_files[0]
                wind_speed_file = wind_speed_files[0]

                # Retrieve wind direction forecast and store in dict
                file_url = f"{self.base_dir(mrs, ha)}/{wind_dir_file}"
                logging.info(f"Downloading {file_url}...")
                response = requests.get(file_url)
                response.raise_for_status()
                data = json.loads(response.text)
                wind_dir_data = data['features']
                # Now trim to items within bbox
                wind_dir_data = [f for f in wind_dir_data
                                 if bbox.contains(f['geometry']['coordinates'][1], f['geometry']['coordinates'][0])]

                # Retrieve wind direction forecast and store in dict
                file_url = f"{self.base_dir(mrs, ha)}/{wind_speed_file}"
                logging.info(f"Downloading {file_url}...")
                response = requests.get(file_url)
                response.raise_for_status()
                data = json.loads(response.text)
                wind_speed_data = data['features']
                # Now trim to items within bbox
                wind_speed_data = [f for f in wind_speed_data
                                   if bbox.contains(f['geometry']['coordinates'][1], f['geometry']['coordinates'][0])]

                # Now we should have 2 dicts populated - one for direction and one for speed.
                # Combine them into wind velocities:
                velocities = self.match_wind_forecasts(wind_speed_data, wind_dir_data, source='ECCC-MSC-Datamart-RDPS')

                all_velocities.extend(velocities)

        return all_velocities


class ECCCAQHIWebscraper(AQHIDataSource):
    region_to_location_id = {
        # Using "Vancouver - NW" as "Vancouver"
        # TODO: expand on below?
        'VANCOUVER': 'JBRIK',
    }
    location_id_to_coordinates = {
        # TODO: Better way to do this ... ? Recall surprising difficulty getting station metadata...
        'JBRIK': [-123.113889, 49.261111],
    }

    def __init__(self):
        config = load_config()
        self.base_url = config["eccc_webscraping_aqhi_observations"]["base_url"]

    def fetch(self, start_date: date, end_date: date, region: str) -> List[AQHI]:
        # TODO: Add error handling? e.g. value DNE, file not found, location ID column not found, ...

        # Get the location ID and coordinates for the given region
        location_id = self.region_to_location_id.get(region.upper())
        longitude, latitude = self.location_id_to_coordinates.get(location_id)

        # Get all months encompassed between start and end dates, inclusive
        months = set()
        current_date = start_date
        while current_date <= end_date:
            months.add((current_date.year, current_date.month))
            current_date += timedelta(days=1)

        # Download relevant month(s) files. There are 3 relevant columns:
        # Date (YYYY-MM-DD), Hour (in UTC - please convert this to PST),
        # and a column named the location ID (e.g. column named JBRIK = JBRIK value).
        # File format is: {self.base_url}/{YYYY}{MM}_MONTHLY_AQHI_PYR_SiteObs_BACKFILLED.csv

        # Loop thru months; download relevant file; build AQHI instances based on observed data from csv
        aqhi_records = []
        for year, month in months:
            file_url = f"{self.base_url}/{year}{month:02d}_MONTHLY_AQHI_PYR_SiteObs_BACKFILLED.csv"
            df = pd.read_csv(file_url)

            # Convert "Date" column to datetime
            df["observed_datetime"] = pd.to_datetime(df["Date"])

            # Convert UTC hour column to proper datetime
            df["observed_datetime"] += pd.to_timedelta(df["Hour (UTC)"], unit="h")

            # Convert to PST (UTC-8) – handle daylight savings
            df["observed_datetime"] = df["observed_datetime"].apply(
                lambda dt: dt.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=-8)))
            )

            # Filter for date range
            df = df[(df["observed_datetime"].dt.date >= start_date) & (df["observed_datetime"].dt.date <= end_date)]

            # Convert to AQHI instances
            for _, row in df.iterrows():
                value = row[location_id]
                if pd.notna(value):
                    aqhi_records.append(
                        AQHI(
                            latitude=latitude,
                            longitude=longitude,
                            observed_datetime=row["observed_datetime"],
                            value=value,
                            source="ECCC-Datamart-AQHI-Monthly"
                        )
                    )

        return aqhi_records


