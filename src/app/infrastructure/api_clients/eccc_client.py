import csv
from io import StringIO
import json
import logging
import pandas as pd
import requests
from typing import List
from datetime import date, datetime, timedelta

from domain.interfaces import WindVelocityDataSource, AQHIDataSource
from domain.entities import WindVelocity, WindVelocityAvg, AQHI
from infrastructure.config import load_config
from infrastructure.persistence.repositories.station_metadata_repo import SQLAlchemyStationRepository
from infrastructure.util.datetime import daterange


class ECCCWindVelocityClient(WindVelocityDataSource):
    stations_metadata_source = SQLAlchemyStationRepository()
    # TODO: better location for above, ideally? Although the data + source are not expected to change...
    region_to_bbox = {
        # Format of a box is: west, south, east, north
        # TODO: expand on below.
        'VANCOUVER': '-123.5,48.5,-122.5,49.5'
    }

    def __init__(self):
        config = load_config()
        self.base_url = config["eccc_msc_geomet_api"]["base_url"]  # config["eccc_webscraping"]["base_url"]
        self.format = config["eccc_msc_geomet_api"]["format"]  # config["eccc_webscraping"]["format"]
        self.lang = config["eccc_msc_geomet_api"]["lang"]

    def fetch(self, start_date: date, end_date: date, region: str = 'VANCOUVER') -> List[WindVelocity]:
        bbox = self.region_to_bbox.get(region)

        # Set URL and params for below GET requests - dates will be added to params below
        data_url = f"{self.base_url}/climate-hourly/items"
        params = {
            'f': {self.format},
            'lang': {self.lang},
            'bbox': bbox,
        }

        # Loop through the days and download data
        all_velocities = []
        for d in daterange(start_date, end_date):
            params.update({
                'LOCAL_YEAR': d.year,
                'LOCAL_MONTH': d.month,
                'LOCAL_DAY': d.day,
            })

            response = requests.get(data_url, params=params)
            if response.status_code == 200:
                data = json.loads(response.content)

                date_velocities = [
                    WindVelocity(
                        latitude=f['geometry']['coordinates'][1],
                        longitude=f['geometry']['coordinates'][0],
                        observed_datetime=datetime.strptime(f['properties']['LOCAL_DATE'], "%Y-%m-%d  %H:%M:%S"),
                        speed=f['properties']['WIND_SPEED'],
                        source_direction=int(f['properties']['WIND_DIRECTION']) * 10,  # To make into degrees clockwise of North
                        source='ECCC-MSC-Geomet-climate-hourly',
                    ) for f in data['features']
                if f['properties']['WIND_SPEED'] is not None and f['properties']['WIND_DIRECTION'] is not None
                ]

                all_velocities.extend(date_velocities)

        return all_velocities

    # TODO_CLEANUP: artifact from webscraping approach
    def fetch_historical_old(self, start_date: date, end_date: date, region: str = 'VANCOUVER') -> List[WindVelocity]:
        bbox = self.region_to_bbox.get(region)
        num_days = (end_date - start_date + timedelta(days=1)).days

        # Get stations within bbox
        stations_in_bbox = self.stations_metadata_source.get_stations(bbox)

        # Set URL and params for below GET requests - the only one that will change is station ID
        data_url = f"{self.base_url}/bulk_data_e.html"
        params = {
            'format': {self.format},
            'Year': end_date.year,
            'Month': end_date.month,
            'Day': end_date.day,
            'timeframe': num_days,
            'submit': 'Download+Data',
        }

        # Loop through the stations and download data
        data = []
        logging.info(f"Downloading data for {len(stations_in_bbox)} stations...")
        for s in stations_in_bbox:
            if s.station_id != 27226:
                pass  # continue  # TODO_DEBUG: temp (only discovery island)

            logging.info(f"Downloading data for station: {s.station_name} (ID: {s.station_id})")

            # Fetch the data
            params['stationID'] = s.station_id
            response = requests.get(data_url, params=params)
            if response.status_code == 200:
                # Response text will be a string. Use StringIO to treat the string as a file object
                csv_reader = csv.DictReader(StringIO(response.text))

                # Discard rows with no wind speed or direction
                station_data = [row for row in csv_reader
                                if row['Wind Dir (10s deg)'] and row['Wind Spd (km/h)']]

                # Convert to velocities
                station_velocities = [
                    WindVelocity(
                        latitude=r['Latitude (y)'],
                        longitude=r['Longitude (x)'],
                        observed_datetime=datetime.strptime(r['Date/Time (LST)'], "%Y-%m-%d  %H:%M"),
                        speed=r['Wind Spd (km/h)'],
                        source_direction=int(r['Wind Dir (10s deg)']) * 10,  # To make into degrees clockwise of North
                        source='ECCC-bulk-data',
                        forecast_datetime=None,
                    )
                    for r in station_data
                ]

                # Now add those velocities to the master list
                if cnt := len(station_velocities):
                    logging.info(f"Found {cnt} velocities for station: {s.station_name} (ID: {s.station_id})")
                    data.extend(station_velocities)
            else:
                logging.info(f"Failed to download data for {s.station_name} (StationID={s.station_id}) (HTTP {response.status_code})")

        return data


class ECCCAQHIForecastClient(AQHIDataSource):
    region_to_bbox = {
        # Format of a box is: west, south, east, north
        # TODO: expand on below.
        'VANCOUVER': '-123.5,48.5,-122.5,49.5'
    }

    def __init__(self):
        config = load_config()
        self.base_url = config["eccc_msc_geomet_api"]["base_url"]
        self.format = config["eccc_msc_geomet_api"]["format"]
        self.lang = config["eccc_msc_geomet_api"]["lang"]
        self.location_id = config["eccc_msc_geomet_api"].get("aqhi_location_id")

    def fetch(self, start_date: date, end_date: date, region: str) -> List[AQHI]:
        # Set URL and params for below GET requests
        data_url = f"{self.base_url}/aqhi-forecasts-realtime/items"
        params = {
            'f': self.format,
            'lang': self.lang,
            'submit': 'Download+Data',
        }
        if self.location_id:
            params.update({'location_id': self.location_id})
        else:
            params.update({'bbox': self.region_to_bbox.get(region)})

        # TODO: add datetime param? Based on start/end dates?

        response = requests.get(data_url, params=params)
        if response.status_code == 200:
            data = json.loads(response.content)
            aqhi_forecasts = [
                AQHI(
                    latitude=f['geometry']['coordinates'][1],
                    longitude=f['geometry']['coordinates'][0],
                    observed_datetime=datetime.fromisoformat(f['properties']['publication_datetime']),
                    value=f['properties']['aqhi'],
                    source='ECCC-MSC-Geomet-aqhi-forecasts-realtime',
                    forecast_datetime=datetime.fromisoformat(f['properties']['forecast_datetime']),
                ) for f in data['features']
            ]

            return aqhi_forecasts

        else:
            return []  # TODO_EH: any valuable messaging in this scenario?

