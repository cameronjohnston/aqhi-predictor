
from datetime import datetime
import uuid

from application.ml.predictors import AQHIPredictor
from domain.entities import ModelTrainingResult
from domain.interfaces import ModelTrainingResultRepository


class PredictFromModelUseCase:
    """Application use case for training an ML model and saving the result"""

    def __init__(self, predictor: AQHIPredictor, model_repository: ModelTrainingResultRepository):
        self.predictor = predictor
        self.model_repository = model_repository

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

        self.model_repository.save(result)
        return result

