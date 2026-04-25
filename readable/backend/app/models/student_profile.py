from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class StudentProfile(Base):
    __tablename__ = "student_profiles"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    reading_level: Mapped[str | None] = mapped_column(nullable=True)
    avg_speed_wpm: Mapped[float] = mapped_column(Float, default=0.0)
    avg_accuracy_pct: Mapped[float] = mapped_column(Float, default=0.0)
    attention_score: Mapped[float] = mapped_column(Float, default=0.0)
    difficult_words: Mapped[list[str]] = mapped_column(JSONB, default=list)
    model_profile_scores: Mapped[dict[str, float]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user = relationship("User", back_populates="student_profile")
