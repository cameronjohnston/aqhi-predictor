from datetime import date
from typing import List, Optional
from sqlalchemy import Column, Float, Integer, String, DateTime, Date
from domain.entities import BBox, WindVelocity, WindVelocityAvg
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

    def get(
            self,
            bbox: Optional[BBox] = None,
            start_date: Optional[date] = None,
            end_date: Optional[date] = None,
            is_forecast: bool = False
    ) -> List[WindVelocity]:
        with session_scope() as session:
            query = session.query(ECCCWindVelocitySQLAlchemyModel)

            if bbox:
                query = query.filter(
                    ECCCWindVelocitySQLAlchemyModel.longitude.between(bbox.west, bbox.east),
                    ECCCWindVelocitySQLAlchemyModel.latitude.between(bbox.south, bbox.north),
                )
            if start_date:
                if is_forecast:
                    query = query.filter(ECCCWindVelocitySQLAlchemyModel.forecast_datetime >= start_date)
                else:
                    query = query.filter(ECCCWindVelocitySQLAlchemyModel.observed_datetime >= start_date)
            if end_date:
                if is_forecast:
                    query = query.filter(ECCCWindVelocitySQLAlchemyModel.forecast_datetime <= end_date)
                else:
                    query = query.filter(ECCCWindVelocitySQLAlchemyModel.observed_datetime <= end_date)
            if is_forecast:
                query = query.filter(ECCCWindVelocitySQLAlchemyModel.forecast_datetime.isnot(None))
            else:
                query = query.filter(ECCCWindVelocitySQLAlchemyModel.forecast_datetime.is_(None))

            # Query DB; build list of wind velocities; return it
            query_res = query.all()
            wind_velocities = [
                WindVelocity(
                    latitude=row.latitude,
                    longitude=row.longitude,
                    observed_datetime=row.observed_datetime,
                    speed=row.speed,
                    source_direction=row.source_direction,
                    source=row.source,
                    forecast_datetime=row.forecast_datetime,
                )
                for row in query_res
            ]
            return wind_velocities


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

    def get(
            self,
            bbox: Optional[BBox] = None,
            start_date: Optional[date] = None,
            end_date: Optional[date] = None,
            is_forecast: bool = False
    ) -> List[WindVelocityAvg]:
        with session_scope() as session:
            query = session.query(ECCCWindVelocityAvgSQLAlchemyModel)

            if bbox:
                query = query.filter(
                    ECCCWindVelocityAvgSQLAlchemyModel.longitude.between(bbox.west, bbox.east),
                    ECCCWindVelocityAvgSQLAlchemyModel.latitude.between(bbox.south, bbox.north),
                )
            if start_date:
                if is_forecast:
                    query = query.filter(ECCCWindVelocityAvgSQLAlchemyModel.forecast_datetime >= start_date)
                else:
                    query = query.filter(ECCCWindVelocityAvgSQLAlchemyModel.observed_date >= start_date)
            if end_date:
                if is_forecast:
                    query = query.filter(ECCCWindVelocityAvgSQLAlchemyModel.forecast_datetime <= end_date)
                else:
                    query = query.filter(ECCCWindVelocityAvgSQLAlchemyModel.observed_date <= end_date)
            if is_forecast:
                query = query.filter(ECCCWindVelocityAvgSQLAlchemyModel.forecast_datetime.isnot(None))
            else:
                query = query.filter(ECCCWindVelocityAvgSQLAlchemyModel.forecast_datetime.is_(None))

            # Query DB; build list of wind velocity avgs; return it
            query_res = query.all()
            wind_velocity_avgs = [
                WindVelocityAvg(
                    latitude=row.latitude,
                    longitude=row.longitude,
                    observed_date=row.observed_date,
                    speed=row.speed,
                    source_direction=row.source_direction,
                    source=row.source,
                    forecast_datetime=row.forecast_datetime,
                )
                for row in query_res
            ]
            return wind_velocity_avgs
