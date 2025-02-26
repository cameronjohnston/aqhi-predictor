from datetime import date
import os
from typing import List, Optional, Tuple
from sqlalchemy import Column, TIMESTAMP, String, JSON
import torch

from domain.entities import ModelTrainingResult
from domain.interfaces import ModelTrainingResultRepository
from infrastructure.persistence.database import Base, session_scope
from infrastructure.config import load_config


class ModelTrainingResultORM(Base):
    """SQLAlchemy ORM model for the table"""
    __tablename__ = "model_training_results"
    __table_args__ = {"schema": "ml"}

    model_id = Column(String, primary_key=True)
    trained_at = Column(TIMESTAMP, nullable=False)
    metrics = Column(JSON, nullable=False)
    status = Column(String, nullable=False)
    model_path = Column(String, nullable=False)  # Stores model file path


class TorchModelFileHandler:
    """Handles saving and loading ML models as files"""
    def __init__(self, model_storage_dir: str):
        self.model_storage_dir = model_storage_dir
        os.makedirs(self.model_storage_dir, exist_ok=True)

    def save_model_data(self, model_training_result: ModelTrainingResult) -> str:
        """Saves model data to disk and returns the file path"""
        model_path = os.path.join(self.model_storage_dir, f"{model_training_result.model_id}.pt")
        torch.save(model_training_result.model_data, model_path)
        return model_path

    def get_model_data(self, model_path: str) -> bytes:
        """Loads model data from a file"""
        return torch.load(model_path)

    def get_latest_model_file_path(self) -> str:
        most_recent_file_full_path, most_recent_time = None, 0

        for entry in os.scandir(self.model_storage_dir):
            if entry.is_file():
                if mod_time := entry.stat().st_mtime_ns > most_recent_time:
                    # update the most recent file and its modification time
                    most_recent_file_full_path = entry.path
                    most_recent_time = mod_time

        # Now we know which file is most recent. Return its full path:
        return most_recent_file_full_path


class PostgresModelTrainingResultRepository(ModelTrainingResultRepository):
    def __init__(self, file_handler_class=TorchModelFileHandler):
        model_storage_dir = load_config()["pytorch_model_training"]["base_dir"]
        self.file_handler = file_handler_class(model_storage_dir=model_storage_dir)

    def save(self, model_training_result: ModelTrainingResult) -> None:
        # 1. Save model data to a file
        full_path = self.file_handler.save_model_data(model_training_result)

        # 2. Save model metadata to postgres
        with session_scope() as session:
            model_orm = ModelTrainingResultORM(
                model_id=model_training_result.model_id,
                trained_at=model_training_result.trained_at,
                metrics=model_training_result.metrics,
                status=model_training_result.status,
                model_path=full_path
            )
            session.add(model_orm)
            session.commit()

    def get(self, model_id: str) -> Optional[ModelTrainingResult]:
        with session_scope() as session:
            model_orm = session.query(ModelTrainingResultORM).filter_by(model_id=model_id).first()
            if not model_orm:
                return None

            # Get model data from path
            model_path = model_orm.model_path
            model_data = self.file_handler.get_model_data(model_path)

            # Now create and return the ModelTrainingResult
            return ModelTrainingResult(
                model_id=model_orm.model_id,
                trained_at=model_orm.trained_at,
                metrics=model_orm.metrics,
                status=model_orm.status,
                model_data=model_data,
            )

    def get_latest(self) -> Optional[ModelTrainingResult]:
        model_path = self.file_handler.get_latest_model_file_path()
        with session_scope() as session:
            model_orm = session.query(ModelTrainingResultORM).filter_by(model_path=model_path).first()
            if not model_orm:
                return None

            # Get model data from path
            model_data = self.file_handler.get_model_data(model_path)

            # Now create and return the ModelTrainingResult
            return ModelTrainingResult(
                model_id=model_orm.model_id,
                trained_at=model_orm.trained_at,
                metrics=model_orm.metrics,
                status=model_orm.status,
                model_data=model_data,
            )

