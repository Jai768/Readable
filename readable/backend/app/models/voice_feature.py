from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class VoiceFeature(Base):
    __tablename__ = "voice_features"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), unique=True, index=True
    )
    speech_rate_wps: Mapped[float] = mapped_column(Float, default=0.0)
    pause_duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    pause_frequency: Mapped[float] = mapped_column(Float, default=0.0)
    mispronunciation_rate: Mapped[float] = mapped_column(Float, default=0.0)
    repetition_rate: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    student = relationship("User")
    session = relationship("Session")
