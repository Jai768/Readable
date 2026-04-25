from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Enum as SqlEnum, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class SessionType(str, Enum):
    diagnostic = "diagnostic"
    reading = "reading"


class SessionStatus(str, Enum):
    active = "active"
    completed = "completed"


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_type: Mapped[SessionType] = mapped_column(SqlEnum(SessionType, name="session_type"))
    status: Mapped[str] = mapped_column(String(50), default=SessionStatus.active.value)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    student = relationship("User", back_populates="sessions", foreign_keys=[student_id])
    result = relationship("SessionResult", back_populates="session", uselist=False)
    progress_entries = relationship("ProgressEntry", back_populates="session")
    eye_tracking_feature = relationship("EyeTrackingFeature", uselist=False)
    voice_feature = relationship("VoiceFeature", uselist=False)
