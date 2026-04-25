from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Enum as SqlEnum, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class UserRole(str, Enum):
    student = "student"
    teacher = "teacher"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(SqlEnum(UserRole, name="user_role"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    student_profile = relationship("StudentProfile", back_populates="user", uselist=False)
    taught_lessons = relationship("Lesson", back_populates="teacher")
    sessions = relationship("Session", back_populates="student")
    personalized_content_items = relationship("PersonalizedContent", back_populates="student")
    progress_entries = relationship("ProgressEntry", back_populates="student")
    eye_tracking_features = relationship("EyeTrackingFeature")
    voice_features = relationship("VoiceFeature")
