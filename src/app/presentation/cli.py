import click
import logging

from application.use_cases.process_wildfire_data import process_wildfire_data
from application.use_cases.process_wind_data import process_wind_velocity_data
from application.use_cases.process_aqhi_data import process_aqhi_data
from application.use_cases.train_and_save_model import TrainAndSaveModelUseCase, TrainModelUseCase
from domain.entities import BBox
from domain.interfaces import (
    WildfireDataSource, WildfireRepository,
    WindVelocityDataSource, WindVelocityRepository, WindVelocityAvgRepository,
    AQHIDataSource, AQHIRepository,
)
from domain.services import WindVelocityService
from infrastructure.api_clients.firms_client import FIRMSClient
from infrastructure.api_clients.eccc_client import ECCCWindVelocityClient, ECCCAQHIForecastClient
from infrastructure.persistence.repositories.aqhi_repo import (
    SQLAlchemyAQHIRepository, SQLAlchemyMLAQHIPredictionRepository
)
from infrastructure.persistence.repositories.wildfire_repo import SQLAlchemyWildfireRepository
from infrastructure.persistence.repositories.wind_velocity_repo import (
    SQLAlchemyWindVelocityRepository, SQLAlchemyWindVelocityAvgRepository
)
from infrastructure.persistence.repositories.model_training_result_repo import PostgresModelTrainingResultRepository
from infrastructure.ml.training.gnn import PyGModelTrainer, PyGTModelTrainer
from infrastructure.ml.predictors import PyGModelPredictor
from infrastructure.webscrapers.eccc_webscraper import ECCCWindVelocityForecastWebscraper, ECCCAQHIWebscraper
from infrastructure.util.logging import setup_logging
from infrastructure.config import load_config


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

        setup_logging(base_dir=load_config()["logging"]["log_dir"],
                      log_file_name=load_config()["logging"]["wildfires_logfile"])

        data_source: WildfireDataSource = FIRMSClient()
        data_target: WildfireRepository = SQLAlchemyWildfireRepository()
        data = process_wildfire_data(data_source, start_date, end_date, region, data_target)
        logging.info(f"Processed {len(data)} wildfires.")

    elif data_type == "wind":

        setup_logging(base_dir=load_config()["logging"]["log_dir"],
                      log_file_name=load_config()["logging"]["wind_logfile"])

        data_source: WindVelocityDataSource = ECCCWindVelocityClient()
        data_target: WindVelocityRepository = SQLAlchemyWindVelocityRepository()
        avg_data_target: WindVelocityAvgRepository = SQLAlchemyWindVelocityAvgRepository()
        velocity_service = WindVelocityService()
        data = process_wind_velocity_data(
            data_source, start_date, end_date, region, data_target,
            # avg_data_target, velocity_service,
        )
        logging.info(f"Processed {len(data)} wind velocity observations.")

    elif data_type == "wind-forecast":

        setup_logging(base_dir=load_config()["logging"]["log_dir"],
                      log_file_name=load_config()["logging"]["wind_forecast_logfile"])

        data_source: WindVelocityDataSource = ECCCWindVelocityForecastWebscraper()
        data_target: WindVelocityRepository = SQLAlchemyWindVelocityRepository()
        data = process_wind_velocity_data(
            data_source, start_date, end_date, region, data_target
        )
        logging.info(f"Processed {len(data)} wind velocity forecasts.")

    elif data_type == "aqhi":

        setup_logging(base_dir=load_config()["logging"]["log_dir"],
                      log_file_name=load_config()["logging"]["aqhi_logfile"])

        data_source: AQHIDataSource = ECCCAQHIWebscraper()
        data_target: AQHIRepository = SQLAlchemyAQHIRepository()
        data = process_aqhi_data(
            data_source, start_date, end_date, region, data_target
        )
        logging.info(f"Processed {len(data)} AQHI data points.")

    elif data_type == "aqhi-forecast":

        setup_logging(base_dir=load_config()["logging"]["log_dir"],
                      log_file_name=load_config()["logging"]["aqhi_forecast_logfile"])

        data_source: AQHIDataSource = ECCCAQHIForecastClient()
        data_target: AQHIRepository = SQLAlchemyAQHIRepository()
        data = process_aqhi_data(
            data_source, start_date, end_date, region, data_target
        )
        logging.info(f"Processed {len(data)} AQHI data points.")


@click.command()
@click.option("--start-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"])
    , help="Start date for fetching wildfires (YYYY-MM-DD).")
@click.option("--end-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"])
    , help="End date for fetching wildfires (YYYY-MM-DD).")
@click.option("--region", default="VANCOUVER", help="Region to fetch data for.")
def train_and_save_model(start_date, end_date, region):
    """CLI command to train the AQHI prediction model."""

    setup_logging(base_dir=load_config()["logging"]["log_dir"],
                  log_file_name=load_config()["logging"]["ml_training_logfile"])

    start_date = start_date.date()
    end_date = end_date.date()
    wildfire_repo = SQLAlchemyWildfireRepository()
    wind_repo = SQLAlchemyWindVelocityRepository()
    aqhi_repo = SQLAlchemyAQHIRepository()

    # Pull from repositories
    logging.info(f"Querying wildfires...")
    wildfires = wildfire_repo.get(start_date=start_date, end_date=end_date, bbox=BBox(-125, 47, -121, 51))
    logging.info(f"Querying wind velocities...")
    wind_velocities = wind_repo.get(start_date=start_date, end_date=end_date)  # TODO: add bbox?
    logging.info(f"Querying AQHI...")
    aqhi_data = aqhi_repo.get(start_date=start_date, end_date=end_date)  # TODO: add bbox?

    model = PyGModelTrainer(wildfires, wind_velocities, aqhi_data, time_intervals=[24])  # TODO: add time intervals
    repository = PostgresModelTrainingResultRepository()
    logging.info(f"Training {model.cn} and saving to {repository.cn}...")

    use_case = TrainAndSaveModelUseCase(
        model_trainer=model,
        repository=repository,
    )
    use_case.execute()
    logging.info(f"Done training {model.cn} and saving to {repository.cn}...")


@click.command()
@click.option("--start-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"])
    , help="Start date for fetching wildfires (YYYY-MM-DD).")
@click.option("--end-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"])
    , help="End date for fetching wildfires (YYYY-MM-DD).")
@click.option("--region", default="VANCOUVER", help="Region to fetch data for.")
def train_model(start_date, end_date, region):
    """CLI command to train the AQHI prediction model."""

    setup_logging(base_dir=load_config()["logging"]["log_dir"],
                  log_file_name=load_config()["logging"]["ml_training_logfile"])

    start_date = start_date.date()
    end_date = end_date.date()
    wildfire_repo = SQLAlchemyWildfireRepository()
    wind_repo = SQLAlchemyWindVelocityRepository()
    aqhi_repo = SQLAlchemyAQHIRepository()

    # Pull from repositories
    logging.info(f"Querying wildfires...")
    wildfires = wildfire_repo.get(start_date=start_date, end_date=end_date, bbox=BBox(-125, 47, -121, 51))
    logging.info(f"Querying wind velocities...")
    wind_velocities = wind_repo.get(start_date=start_date, end_date=end_date)  # TODO: add bbox?
    logging.info(f"Querying AQHI...")
    aqhi_data = aqhi_repo.get(start_date=start_date, end_date=end_date)  # TODO: add bbox?

    # Trim to only noon observations
    wildfires = [wf for wf in wildfires if wf.observed_datetime.hour == 12]
    wind_velocities = [wv for wv in wind_velocities if wv.observed_datetime.hour == 12]
    aqhi_data = [aq for aq in aqhi_data if aq.observed_datetime.hour == 12]

    logging.info(f"Training from {len(wildfires)} fires, {len(wind_velocities)} wind, {len(aqhi_data)} AQ")

    model_trainer = PyGTModelTrainer(wildfires, wind_velocities, aqhi_data)  # TODO: add time intervals
    logging.info(f"Training {model_trainer.cn}...")

    use_case = TrainModelUseCase(
        model_trainer=model_trainer,
    )
    use_case.execute()
    logging.info(f"Done training {model_trainer.cn}.")


@click.command()
@click.option("--start-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"])
    , help="Start date for fetching wildfires (YYYY-MM-DD).")
@click.option("--end-date", required=True, type=click.DateTime(formats=["%Y-%m-%d"])
    , help="End date for fetching wildfires (YYYY-MM-DD).")
@click.option("--region", default="VANCOUVER", help="Region to fetch data for.")
def predict_from_model(start_date, end_date, region):
    """CLI command to predict AQHI based on the AQHI prediction model."""

    setup_logging(base_dir=load_config()["logging"]["log_dir"],
                  log_file_name=load_config()["logging"]["ml_predicting_logfile"])

    start_date = start_date.date()
    end_date = end_date.date()
    wildfire_repo = SQLAlchemyWildfireRepository()
    wind_repo = SQLAlchemyWindVelocityRepository()
    aqhi_repo = SQLAlchemyAQHIRepository()
    model_repo = PostgresModelTrainingResultRepository()
    prediction_result_repo = SQLAlchemyMLAQHIPredictionRepository()

    # Pull from repositories
    logging.info(f"Querying wildfires...")
    wildfires = wildfire_repo.get(start_date=start_date, end_date=end_date, bbox=BBox(-125, 47, -121, 51))
    logging.info(f"Querying wind velocities...")
    wind_velocities = wind_repo.get(start_date=start_date, end_date=end_date)  # TODO: add bbox?
    logging.info(f"Querying AQHI...")
    aqhi_data = aqhi_repo.get(start_date=start_date, end_date=end_date)  # TODO: add bbox?
    logging.info(f"Querying latest model training result...")
    model_training_result = model_repo.get_latest()

    predictor = PyGModelPredictor(
        wildfires=wildfires,
        wind_velocities=wind_velocities,
        aqhi_data=aqhi_data,
        time_intervals=[24],  # Use same intervals as training
        model_training_result=model_training_result
    )  # TODO: add time intervals

    predicted_aqhi = predictor.predict()
    logging.info(f"Got {len(predicted_aqhi)} AQHI predictions.")

    # Save to AQHI repo
    prediction_result_repo.save(predicted_aqhi)

