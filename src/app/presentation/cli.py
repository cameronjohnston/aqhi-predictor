import click

from application.use_cases.process_wildfire_data import process_wildfire_data
from application.use_cases.process_wind_data import process_wind_velocity_data
from application.use_cases.process_aqhi_data import process_aqhi_data
from domain.interfaces import (
    WildfireDataSource, WildfireRepository,
    WindVelocityDataSource, WindVelocityRepository, WindVelocityAvgRepository,
    AQHIDataSource, AQHIRepository,
)
from domain.services import WindVelocityService
from infrastructure.api_clients.firms_client import FIRMSClient
from infrastructure.api_clients.eccc_client import ECCCWindVelocityClient, ECCCAQHIForecastClient
from infrastructure.persistence.repositories.aqhi_repo import SQLAlchemyAQHIRepository
from infrastructure.persistence.repositories.wildfire_repo import SQLAlchemyWildfireRepository
from infrastructure.persistence.repositories.wind_velocity_repo import (
    SQLAlchemyWindVelocityRepository, SQLAlchemyWindVelocityAvgRepository
)
from infrastructure.ml.training.gnn import PyGModelTrainer
from infrastructure.webscrapers.eccc_webscraper import ECCCWindVelocityForecastWebscraper, ECCCAQHIWebscraper


@click.command()
@click.option(
    "--data-type", type=click.Choice(["wildfire", "wind", "wind-forecast", "aqhi", "aqhi-forecast"], case_sensitive=False), required=True,
    help="Type of data to fetch: 'wildfire' or 'wind' or 'wind-forecast' or 'aqhi' or 'aqhi-forecast'.",
)
@click.option("--start-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"])
    , help="Start date for fetching data (YYYY-MM-DD).")
@click.option("--end-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"])
    , help="End date for fetching data (YYYY-MM-DD).")
@click.option("--region", default="VANCOUVER", help="Region to fetch data for.")
def fetch_data(data_type, start_date, end_date, region):
    start_date = start_date.date()
    end_date = end_date.date()

    if data_type == "wildfire":
        data_source: WildfireDataSource = FIRMSClient()
        data_target: WildfireRepository = SQLAlchemyWildfireRepository()
        data = process_wildfire_data(data_source, start_date, end_date, region, data_target)
        print(f"Processed {len(data)} wildfires.")

    elif data_type == "wind":
        data_source: WindVelocityDataSource = ECCCWindVelocityClient()
        data_target: WindVelocityRepository = SQLAlchemyWindVelocityRepository()
        avg_data_target: WindVelocityAvgRepository = SQLAlchemyWindVelocityAvgRepository()
        velocity_service = WindVelocityService()
        data = process_wind_velocity_data(
            data_source, start_date, end_date, region, data_target,
            avg_data_target, velocity_service
        )
        print(f"Processed {len(data)} wind velocity observations.")

    elif data_type == "wind-forecast":
        data_source: WindVelocityDataSource = ECCCWindVelocityForecastWebscraper()
        data_target: WindVelocityRepository = SQLAlchemyWindVelocityRepository()
        data = process_wind_velocity_data(
            data_source, start_date, end_date, region, data_target
        )
        print(f"Processed {len(data)} wind velocity forecasts.")

    elif data_type == "aqhi":
        data_source: AQHIDataSource = ECCCAQHIWebscraper()
        data_target: AQHIRepository = SQLAlchemyAQHIRepository()
        data = process_aqhi_data(
            data_source, start_date, end_date, region, data_target
        )
        print(f"Processed {len(data)} AQHI data points.")

    elif data_type == "aqhi-forecast":
        data_source: AQHIDataSource = ECCCAQHIForecastClient()
        data_target: AQHIRepository = SQLAlchemyAQHIRepository()
        data = process_aqhi_data(
            data_source, start_date, end_date, region, data_target
        )
        print(f"Processed {len(data)} AQHI data points.")


@click.command()
@click.option("--start-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"])
    , help="Start date for fetching wildfires (YYYY-MM-DD).")
@click.option("--end-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"])
    , help="End date for fetching wildfires (YYYY-MM-DD).")
@click.option("--region", default="VANCOUVER", help="Region to fetch data for.")
def train_model(start_date, end_date, region):
    """CLI command to train the AQHI prediction model."""
    start_date = start_date.date()
    end_date = end_date.date()
    wildfire_repo = SQLAlchemyWildfireRepository()
    wind_repo = SQLAlchemyWindVelocityRepository()
    aqhi_repo = SQLAlchemyAQHIRepository()

    # Pull from repositories
    print(f"Querying wildfires...")
    wildfires = wildfire_repo.get(start_date=start_date, end_date=end_date)  # TODO: add bbox?
    print(f"Querying wind velocities...")
    wind_velocities = wind_repo.get(start_date=start_date, end_date=end_date)  # TODO: add bbox?
    print(f"Querying AQHI...")
    aqhi_data = aqhi_repo.get(start_date=start_date, end_date=end_date)  # TODO: add bbox?

    model = PyGModelTrainer(wildfires, wind_velocities, aqhi_data, time_intervals=[24])  # TODO: add time intervals
    print(f"Preparing training data in {model.cn}...")
    model.prepare_training_data()

