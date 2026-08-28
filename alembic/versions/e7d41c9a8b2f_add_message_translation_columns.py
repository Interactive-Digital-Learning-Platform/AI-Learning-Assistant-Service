"""add message translation columns

Revision ID: e7d41c9a8b2f
Revises: bc262c6604ac
Create Date: 2026-08-28 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e7d41c9a8b2f"
down_revision: Union[str, Sequence[str], None] = "bc262c6604ac"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column(
            "is_translated",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "messages",
        sa.Column("translated_content", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("messages", "translated_content")
    op.drop_column("messages", "is_translated")
