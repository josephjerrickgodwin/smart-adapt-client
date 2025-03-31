import os
from contextlib import contextmanager
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Define the db path and connection string
DATA_DIR = os.path.join(os.getcwd(), 'database')
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DATA_DIR}/data.db")

# Define the DB engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Create a Local Session
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)
metadata_obj = MetaData(schema=None)
Base = declarative_base(metadata=metadata_obj)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


get_db = contextmanager(get_session)
