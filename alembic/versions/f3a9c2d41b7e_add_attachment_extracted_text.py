"""add attachment extracted text

Revision ID: f3a9c2d41b7e
Revises: e7d41c9a8b2f
Create Date: 2026-10-17 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f3a9c2d41b7e"
down_revision: Union[str, Sequence[str], None] = "e7d41c9a8b2f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "attachments",
        sa.Column("extracted_text", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("attachments", "extracted_text")
