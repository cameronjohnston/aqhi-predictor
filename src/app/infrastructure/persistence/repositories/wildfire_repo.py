from datetime import date
from typing import List, Optional
from sqlalchemy import Column, Date, String, Float
from domain.entities import BBox, Wildfire
from domain.interfaces import WildfireRepository
from infrastructure.persistence.database import Base, session_scope


class FIRMSWildfireSQLAlchemyModel(Base):
    __tablename__ = "wildfires"
    __table_args__ = {"schema": "firms"}

    id = Column(String, primary_key=True)
    first_discovered = Column(Date, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    frp = Column(Float, nullable=False)  # Fire Radiative Power
    asofdate = Column(Date, nullable=False)
    source = Column(String, nullable=False)


class SQLAlchemyWildfireRepository(WildfireRepository):
    def save(self, wildfires: List[Wildfire]) -> None:
        wildfire_models = [
            FIRMSWildfireSQLAlchemyModel(
                id=w.id,
                first_discovered=w.first_discovered,
                latitude=w.latitude,
                longitude=w.longitude,
                frp=w.frp,
                asofdate=w.asofdate,
                source=w.source,
            )
            for w in wildfires
        ]

        with session_scope() as session:
            # TODO_EH: what if the insert fails? e.g. PK conflict, non-nullable column missing, value to large, ...
            session.add_all(wildfire_models)
            session.commit()

    def get(
            self,
            bbox: Optional[BBox] = None,
            start_date: Optional[date] = None,
            end_date: Optional[date] = None,
            source: Optional[str] = None,
    ) -> List[Wildfire]:
        with session_scope() as session:
            query = session.query(FIRMSWildfireSQLAlchemyModel)

            if bbox:
                query = query.filter(
                    FIRMSWildfireSQLAlchemyModel.longitude.between(bbox.west, bbox.east),
                    FIRMSWildfireSQLAlchemyModel.latitude.between(bbox.south, bbox.north),
                )
            if start_date:
                query = query.filter(FIRMSWildfireSQLAlchemyModel.asofdate >= start_date)
            if end_date:
                query = query.filter(FIRMSWildfireSQLAlchemyModel.asofdate <= end_date)
            if source:
                query = query.filter(FIRMSWildfireSQLAlchemyModel.source == source)

            # Query DB; build list of wildfires; return it
            query_res = query.all()
            wildfires = [
                Wildfire(
                    id=row.id,
                    first_discovered=row.first_discovered,
                    latitude=row.latitude,
                    longitude=row.longitude,
                    frp=row.frp,
                    asofdate=row.asofdate,
                    source=row.source,
                )
                for row in query_res
            ]
            return wildfires
