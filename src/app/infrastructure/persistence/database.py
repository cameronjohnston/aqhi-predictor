from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from infrastructure.config import load_config


# Singleton engine and Base
_engine = None
Base = declarative_base()


def get_engine():
    global _engine
    if _engine is None:
        config = load_config()
        database_url = config["database"]["url"]
        _engine = create_engine(database_url)
    return _engine


def get_session():
    engine = get_engine()
    return sessionmaker(bind=engine)()


# Context manager to handle session lifecycle
@contextmanager
def session_scope():
    session = get_session()  # Create a new session
    try:
        yield session  # Allow code to use the session
        session.commit()  # Commit transaction if no exceptions
    except Exception:
        session.rollback()  # Rollback in case of an error
        raise  # Re-raise the exception
    finally:
        session.close()  # Ensure the session is closed

