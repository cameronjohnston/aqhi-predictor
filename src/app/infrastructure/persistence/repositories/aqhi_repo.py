from datetime import date
from typing import List, Optional
from sqlalchemy import Column, Float, Integer, String, DateTime, Date, and_
from domain.entities import AQHI, BBox
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


class SQLAlchemyAQHIRepository(AQHIRepository):
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

    def get(
        self,
        bbox: Optional[BBox] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        is_forecast: bool = False,
    ) -> List[AQHI]:
        with session_scope() as session:
            query = session.query(ECCCAQHISQLAlchemyModel)

            if bbox:
                query = query.filter(
                    ECCCAQHISQLAlchemyModel.longitude.between(bbox.west, bbox.east),
                    ECCCAQHISQLAlchemyModel.latitude.between(bbox.south, bbox.north),
                )
            if start_date:
                if is_forecast:
                    query = query.filter(ECCCAQHISQLAlchemyModel.forecast_datetime >= start_date)
                else:
                    query = query.filter(ECCCAQHISQLAlchemyModel.observed_datetime >= start_date)
            if end_date:
                if is_forecast:
                    query = query.filter(ECCCAQHISQLAlchemyModel.forecast_datetime <= end_date)
                else:
                    query = query.filter(ECCCAQHISQLAlchemyModel.observed_datetime <= end_date)
            if is_forecast:
                query = query.filter(ECCCAQHISQLAlchemyModel.forecast_datetime.isnot(None))
            else:
                query = query.filter(ECCCAQHISQLAlchemyModel.forecast_datetime.is_(None))

            # Query DB; build list of AQHI instances; return it
            query_res = query.all()
            aqhi_data = [
                AQHI(
                    latitude=row.latitude,
                    longitude=row.longitude,
                    observed_datetime=row.observed_datetime,
                    value=row.value,
                    source=row.source,
                    forecast_datetime=row.forecast_datetime,
                )
                for row in query_res
            ]
            return aqhi_data


class MLAQHIPredictionSQLAlchemyModel(Base):
    __tablename__ = "aqhi_prediction"
    __table_args__ = {"schema": "ml"}

    latitude = Column(Float, primary_key=True, nullable=False)
    longitude = Column(Float, primary_key=True, nullable=False)
    observed_datetime = Column(DateTime, primary_key=True, nullable=False)
    value = Column(Float, nullable=False)
    source = Column(String, nullable=False)
    forecast_datetime = Column(DateTime, nullable=True)


class SQLAlchemyMLAQHIPredictionRepository(AQHIRepository):
    def save(self, aqhi_data: List[AQHI]) -> None:
        aqhi_models = [
            MLAQHIPredictionSQLAlchemyModel(
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

    def get(
        self,
        bbox: Optional[BBox] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        is_forecast: bool = False,
    ) -> List[AQHI]:
        with session_scope() as session:
            query = session.query(MLAQHIPredictionSQLAlchemyModel)

            if bbox:
                query = query.filter(
                    MLAQHIPredictionSQLAlchemyModel.longitude.between(bbox.west, bbox.east),
                    MLAQHIPredictionSQLAlchemyModel.latitude.between(bbox.south, bbox.north),
                )
            if start_date:
                if is_forecast:
                    query = query.filter(MLAQHIPredictionSQLAlchemyModel.forecast_datetime >= start_date)
                else:
                    query = query.filter(MLAQHIPredictionSQLAlchemyModel.observed_datetime >= start_date)
            if end_date:
                if is_forecast:
                    query = query.filter(MLAQHIPredictionSQLAlchemyModel.forecast_datetime <= end_date)
                else:
                    query = query.filter(MLAQHIPredictionSQLAlchemyModel.observed_datetime <= end_date)
            if is_forecast:
                query = query.filter(MLAQHIPredictionSQLAlchemyModel.forecast_datetime.isnot(None))
            else:
                query = query.filter(MLAQHIPredictionSQLAlchemyModel.forecast_datetime.is_(None))

            # Query DB; build list of AQHI instances; return it
            query_res = query.all()
            aqhi_data = [
                AQHI(
                    latitude=row.latitude,
                    longitude=row.longitude,
                    observed_datetime=row.observed_datetime,
                    value=row.value,
                    source=row.source,
                    forecast_datetime=row.forecast_datetime,
                )
                for row in query_res
            ]
            return aqhi_data


