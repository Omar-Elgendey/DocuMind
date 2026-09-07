"""Database initialization script to create tables in MySQL."""

from app.db.session import Base, engine

# Import models so SQLAlchemy registers them with Base.metadata.
import app.db.models  # noqa: F401


def init_db() -> None:
    print("Creating database tables in MySQL...")

    Base.metadata.create_all(bind=engine)

    print("Database tables created successfully!")


if __name__ == "__main__":
    init_db()

