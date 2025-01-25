from typing import List, Optional
from sqlalchemy import Column, Float, Integer, String, and_
from domain.models import StationMetadata
from infrastructure.persistence.database import Base, session_scope


class PCICStationSQLAlchemyModel(Base):
    __tablename__ = "station"
    __table_args__ = {"schema": "pcic"}

    network_id = Column(Integer, nullable=False)
    network_name = Column(String, nullable=False)
    station_id = Column(Integer, primary_key=True, nullable=False)
    station_name = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float, nullable=True)
    source = Column(String, nullable=False)


class ECCCWeatherStationSQLAlchemyModel(Base):
    __tablename__ = "weather_station"
    __table_args__ = {"schema": "eccc"}

    station_id = Column(Integer, primary_key=True, nullable=False)
    station_name = Column(String, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float, nullable=True)
    source = Column(String, nullable=False)


class SQLAlchemyStationRepository:
    def save(self, stations: List[StationMetadata]) -> None:
        station_models = [
            ECCCWeatherStationSQLAlchemyModel(
                station_id=s.station_id,
                station_name=s.station_name,
                latitude=s.latitude,
                longitude=s.longitude,
                elevation=s.elevation,
                source=s.source,
            )
            for s in stations
        ]

        with session_scope() as session:
            session.add_all(station_models)
            session.commit()

    def get_stations(self, bbox: Optional[str] = None) -> List[StationMetadata]:
        """
        Retrieves stations from the database. If a bbox is provided, filters by bounding box.

        :param bbox: Optional comma-delimited string "west,south,east,north"
        :return: List of StationMetadata objects
        """
        with session_scope() as session:
            query = session.query(ECCCWeatherStationSQLAlchemyModel)

            if bbox:
                try:
                    west, south, east, north = map(float, bbox.split(","))
                    query = query.filter(
                        and_(
                            ECCCWeatherStationSQLAlchemyModel.longitude >= west,
                            ECCCWeatherStationSQLAlchemyModel.longitude <= east,
                            ECCCWeatherStationSQLAlchemyModel.latitude >= south,
                            ECCCWeatherStationSQLAlchemyModel.latitude <= north,
                        )
                    )
                except ValueError:
                    raise ValueError("Invalid bbox format. Expected 'west,south,east,north'.")

            stations = query.all()

            return [
                StationMetadata(
                    station_id=s.station_id,
                    station_name=s.station_name,
                    latitude=s.latitude,
                    longitude=s.longitude,
                    elevation=s.elevation,
                    source=s.source,
                )
                for s in stations
            ]
