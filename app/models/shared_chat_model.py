import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SharedChat(Base):
    """A read-only snapshot of a conversation, reachable by its share token.

    One link per conversation: sharing again refreshes the snapshot and keeps the
    token. The snapshot never stores conversation or message ids, so a viewer
    can't use it to reach the live conversation.
    """

    __tablename__ = "shared_chats"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    token: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    title: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # list[SharedMessage] as JSON (see app/schemas/share.py)
    messages: Mapped[list] = mapped_column(JSONB, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
