from typing import List
from sqlalchemy import Column, Date, String, Float
from domain.models import Wildfire
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
