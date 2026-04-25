from datetime import datetime

from pydantic import BaseModel


class SessionSummary(BaseModel):
    session_id: int
    session_type: str
    status: str
    started_at: datetime
    ended_at: datetime | None
    accuracy_pct: float | None = None
    speed_wpm: float | None = None


class StudentProfileResponse(BaseModel):
    student_id: int
    email: str
    reading_level: str | None
    avg_speed_wpm: float
    avg_accuracy_pct: float
    attention_score: float
    difficult_words: list[str]
    model_profile_scores: dict[str, float]
    recent_sessions: list[SessionSummary]


class ProgressEntryResponse(BaseModel):
    id: int
    session_id: int
    accuracy_trend: float
    words_practiced: list[str]
    timestamp: datetime


class StudentProgressResponse(BaseModel):
    student_id: int
    entries: list[ProgressEntryResponse]
    difficult_words: list[str]
