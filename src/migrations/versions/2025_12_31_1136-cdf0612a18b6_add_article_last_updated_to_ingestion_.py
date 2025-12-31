"""add_article_last_updated_to_ingestion_progress

Revision ID: cdf0612a18b6
Revises: fc8d7cb9e3ec
Create Date: 2025-12-31 11:36:31.536243

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "cdf0612a18b6"
down_revision: str | None = "fc8d7cb9e3ec"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add article_last_updated column for change detection during re-ingestion
    op.add_column(
        "ingestion_progress", sa.Column("article_last_updated", sa.String(50), nullable=True)
    )
    # Create index on article_last_updated for efficient change detection queries
    op.create_index(
        "ix_ingestion_progress_article_last_updated", "ingestion_progress", ["article_last_updated"]
    )


def downgrade() -> None:
    op.drop_index("ix_ingestion_progress_article_last_updated", table_name="ingestion_progress")
    op.drop_column("ingestion_progress", "article_last_updated")
