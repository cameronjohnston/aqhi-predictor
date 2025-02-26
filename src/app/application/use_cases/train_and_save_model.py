
from datetime import datetime
import uuid

from application.ml.trainers import ModelTrainer
from domain.entities import ModelTrainingResult
from domain.interfaces import ModelTrainingResultRepository


class TrainAndSaveModelUseCase:
    """Application use case for training an ML model and saving the result"""

    def __init__(self, model_trainer: ModelTrainer, repository: ModelTrainingResultRepository):
        self.model_trainer = model_trainer
        self.repository = repository

    def execute(self) -> ModelTrainingResult:
        """Executes model training and persists the result"""

        model, metrics = self.model_trainer.train()  # Train model
        model_id = str(uuid.uuid4())

        result = ModelTrainingResult(
            model_id=model_id,
            trained_at=datetime.now(),
            metrics=metrics,
            status="completed",
            model_data=model.state_dict()  # Get model parameters
        )

        self.repository.save(result)
        return result


class TrainModelUseCase:
    """Application use case for training an ML model and not saving the result"""

    def __init__(self, model_trainer: ModelTrainer):
        self.model_trainer = model_trainer

    def execute(self) -> None:
        """Executes model training and does not persist the result"""

        self.model_trainer.train()  # Train model
