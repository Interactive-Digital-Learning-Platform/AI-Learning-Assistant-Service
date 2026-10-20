"""add shared chats table

Revision ID: a4c8e2f61d93
Revises: f3a9c2d41b7e
Create Date: 2026-10-19 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "a4c8e2f61d93"
down_revision: str | Sequence[str] | None = "f3a9c2d41b7e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "shared_chats",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("token", sa.String(length=64), nullable=False),
        sa.Column("conversation_id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("messages", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id"),
    )
    op.create_index(op.f("ix_shared_chats_token"), "shared_chats", ["token"], unique=True)
    op.create_index(op.f("ix_shared_chats_user_id"), "shared_chats", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_shared_chats_user_id"), table_name="shared_chats")
    op.drop_index(op.f("ix_shared_chats_token"), table_name="shared_chats")
    op.drop_table("shared_chats")
