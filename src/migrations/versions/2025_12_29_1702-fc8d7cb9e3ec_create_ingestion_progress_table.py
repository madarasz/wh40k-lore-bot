"""create_ingestion_progress_table

Revision ID: fc8d7cb9e3ec
Revises: 2ac2f41edf9c
Create Date: 2025-12-29 17:02:36.193589

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fc8d7cb9e3ec"
down_revision: str | None = "2ac2f41edf9c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ingestion_progress",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("article_id", sa.String(100), nullable=False, index=True),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("batch_number", sa.Integer, nullable=False),
        sa.Column("processed_at", sa.DateTime, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    # Create index on article_id for faster lookups
    op.create_index("idx_ingestion_progress_article_id", "ingestion_progress", ["article_id"])
    # Create index on status for filtering incomplete batches
    op.create_index("idx_ingestion_progress_status", "ingestion_progress", ["status"])


def downgrade() -> None:
    op.drop_index("idx_ingestion_progress_status", table_name="ingestion_progress")
    op.drop_index("idx_ingestion_progress_article_id", table_name="ingestion_progress")
    op.drop_table("ingestion_progress")
