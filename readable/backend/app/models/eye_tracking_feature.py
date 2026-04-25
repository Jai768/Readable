from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class EyeTrackingFeature(Base):
    __tablename__ = "eye_tracking_features"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), unique=True, index=True
    )
    fixation_duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    saccade_length: Mapped[float] = mapped_column(Float, default=0.0)
    regression_count: Mapped[int] = mapped_column(Integer, default=0)
    skipped_words: Mapped[int] = mapped_column(Integer, default=0)
    reading_speed_wpm: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    student = relationship("User")
    session = relationship("Session")
