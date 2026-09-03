"""
database.py — SQLAlchemy engine, session factory, declarative Base, and
the get_db FastAPI dependency.

Usage:
    from database import Base, get_db

    # In a route:
    def my_route(db: Session = Depends(get_db)):
        ...
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,   # reconnect automatically if connection drops
    echo=settings.db_echo,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """All ORM models inherit from this base."""
    pass


def get_db():
    """FastAPI dependency that yields a database session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
