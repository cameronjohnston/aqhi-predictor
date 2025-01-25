from typing import List
from sqlalchemy import Column, Float, Integer, String, DateTime, Date
from domain.models import AQHI
from domain.interfaces import AQHIRepository
from infrastructure.persistence.database import Base, session_scope


class ECCCAQHISQLAlchemyModel(Base):
    __tablename__ = "aqhi"
    __table_args__ = {"schema": "eccc"}

    latitude = Column(Float, primary_key=True, nullable=False)
    longitude = Column(Float, primary_key=True, nullable=False)
    observed_datetime = Column(DateTime, primary_key=True, nullable=False)
    value = Column(Float, nullable=False)
    source = Column(String, nullable=False)
    forecast_datetime = Column(DateTime, nullable=True)


class SQLAlchemyAQHIRepository:
    def save(self, aqhi_data: List[AQHI]) -> None:
        aqhi_models = [
            ECCCAQHISQLAlchemyModel(
                latitude=a.latitude,
                longitude=a.longitude,
                observed_datetime=a.observed_datetime,
                value=a.value,
                source=a.source,
                forecast_datetime=a.forecast_datetime,
            )
            for a in aqhi_data
        ]

        with session_scope() as session:
            session.add_all(aqhi_models)
            session.commit()



