"""Pytest configuration and shared fixtures."""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.models.base import Base


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Return path to temporary test database."""
    return tmp_path / "test.db"


@pytest.fixture
async def async_engine(temp_db_path: Path):
    """Create async engine for testing and run migrations."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{temp_db_path}", echo=False)

    # Run Alembic migrations to create tables with all indexes
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{temp_db_path}")
    command.upgrade(alembic_cfg, "head")

    yield engine

    # Drop all tables after test
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async session for testing."""
    async_session_maker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()
