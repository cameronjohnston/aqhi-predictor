from typing import List
from sqlalchemy import Column, Float, Integer, String, DateTime, Date
from domain.models import WindVelocity, WindVelocityAvg
from domain.interfaces import WindVelocityRepository, WindVelocityAvgRepository
from infrastructure.persistence.database import Base, session_scope


class ECCCWindVelocitySQLAlchemyModel(Base):
    __tablename__ = "wind_velocity"
    __table_args__ = {"schema": "eccc"}

    latitude = Column(Float, primary_key=True, nullable=False)
    longitude = Column(Float, primary_key=True, nullable=False)
    observed_datetime = Column(DateTime, primary_key=True, nullable=False)
    speed = Column(Float, nullable=False)
    source_direction = Column(Integer, nullable=False)
    source = Column(String, nullable=False)
    forecast_datetime = Column(DateTime, nullable=True)


class SQLAlchemyWindVelocityRepository(WindVelocityRepository):
    def save(self, wind_velocities: List[WindVelocity]) -> None:
        wind_velocity_models = [
            ECCCWindVelocitySQLAlchemyModel(
                latitude=w.latitude,
                longitude=w.longitude,
                observed_datetime=w.observed_datetime,
                speed=w.speed,
                source_direction=w.source_direction,
                source=w.source,
                forecast_datetime=w.forecast_datetime,
            )
            for w in wind_velocities
        ]

        with session_scope() as session:
            session.add_all(wind_velocity_models)
            session.commit()


class ECCCWindVelocityAvgSQLAlchemyModel(Base):
    __tablename__ = "wind_velocity_avg"
    __table_args__ = {"schema": "eccc"}

    latitude = Column(Float, primary_key=True, nullable=False)
    longitude = Column(Float, primary_key=True, nullable=False)
    observed_date = Column(Date, primary_key=True, nullable=False)
    speed = Column(Float, nullable=False)
    source_direction = Column(Integer, nullable=False)
    source = Column(String, nullable=False)
    forecast_datetime = Column(DateTime, nullable=True)


class SQLAlchemyWindVelocityAvgRepository(WindVelocityAvgRepository):
    def save(self, wind_velocity_avgs: List[WindVelocityAvg]) -> None:
        wind_velocity_avg_models = [
            ECCCWindVelocityAvgSQLAlchemyModel(
                latitude=w.latitude,
                longitude=w.longitude,
                observed_date=w.observed_date,
                speed=w.speed,
                source_direction=w.source_direction,
                source=w.source,
                forecast_datetime=w.forecast_datetime,
            )
            for w in wind_velocity_avgs
        ]

        with session_scope() as session:
            session.add_all(wind_velocity_avg_models)
            session.commit()
