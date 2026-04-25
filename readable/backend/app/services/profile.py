from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ProgressEntry, Session, SessionResult, StudentProfile, User
from app.schemas.profile import (
    ProgressEntryResponse,
    SessionSummary,
    StudentProfileResponse,
    StudentProgressResponse,
)


async def create_or_update(
    db: AsyncSession,
    student_id: int,
    results: dict[str, object],
) -> StudentProfile:
    profile_result = await db.execute(
        select(StudentProfile).where(StudentProfile.user_id == student_id)
    )
    profile = profile_result.scalar_one_or_none()

    errors = results.get("errors", [])
    error_words = [str(item["word"]) for item in errors if isinstance(item, dict) and item.get("word")]
    difficulty_pool = set(error_words)
    if profile is None:
        profile = StudentProfile(
            user_id=student_id,
            reading_level=_reading_level(float(results["accuracy_pct"])),
            avg_speed_wpm=float(results["speed_wpm"]),
            avg_accuracy_pct=float(results["accuracy_pct"]),
            attention_score=float(results["attention_score"]),
            difficult_words=sorted(difficulty_pool),
            model_profile_scores=_extract_model_profile_scores(results),
        )
        db.add(profile)
        await db.flush()
        return profile

    profile.avg_speed_wpm = round((profile.avg_speed_wpm + float(results["speed_wpm"])) / 2, 2)
    profile.avg_accuracy_pct = round(
        (profile.avg_accuracy_pct + float(results["accuracy_pct"])) / 2, 2
    )
    profile.attention_score = round(
        (profile.attention_score + float(results["attention_score"])) / 2, 2
    )
    profile.reading_level = _reading_level(profile.avg_accuracy_pct)
    profile.difficult_words = sorted(set(profile.difficult_words).union(difficulty_pool))
    model_scores = _extract_model_profile_scores(results)
    if model_scores:
        profile.model_profile_scores = model_scores
    await db.flush()
    return profile


async def build_profile_response(db: AsyncSession, student_id: int) -> StudentProfileResponse:
    user_result = await db.execute(select(User).where(User.id == student_id))
    user = user_result.scalar_one_or_none()
    if user is None:
        raise ValueError("Student not found")

    profile_result = await db.execute(
        select(StudentProfile).where(StudentProfile.user_id == student_id)
    )
    profile = profile_result.scalar_one_or_none()
    if profile is None:
        profile = StudentProfile(
            user_id=student_id,
            reading_level=None,
            avg_speed_wpm=0.0,
            avg_accuracy_pct=0.0,
            attention_score=0.0,
            difficult_words=[],
            model_profile_scores={},
        )

    sessions_result = await db.execute(
        select(Session, SessionResult)
        .outerjoin(SessionResult, SessionResult.session_id == Session.id)
        .where(Session.student_id == student_id)
        .order_by(desc(Session.started_at))
        .limit(5)
    )

    recent_sessions = [
        SessionSummary(
            session_id=session.id,
            session_type=session.session_type.value
            if hasattr(session.session_type, "value")
            else str(session.session_type),
            status=session.status,
            started_at=session.started_at,
            ended_at=session.ended_at,
            accuracy_pct=result.accuracy_pct if result else None,
            speed_wpm=result.speed_wpm if result else None,
        )
        for session, result in sessions_result.all()
    ]

    return StudentProfileResponse(
        student_id=student_id,
        email=user.email,
        reading_level=profile.reading_level,
        avg_speed_wpm=profile.avg_speed_wpm,
        avg_accuracy_pct=profile.avg_accuracy_pct,
        attention_score=profile.attention_score,
        difficult_words=profile.difficult_words,
        model_profile_scores=profile.model_profile_scores,
        recent_sessions=recent_sessions,
    )


async def build_progress_response(db: AsyncSession, student_id: int) -> StudentProgressResponse:
    profile_result = await db.execute(
        select(StudentProfile).where(StudentProfile.user_id == student_id)
    )
    profile = profile_result.scalar_one_or_none()

    entries_result = await db.execute(
        select(ProgressEntry)
        .where(ProgressEntry.student_id == student_id)
        .order_by(desc(ProgressEntry.timestamp))
        .limit(10)
    )
    entries = entries_result.scalars().all()

    return StudentProgressResponse(
        student_id=student_id,
        entries=[
            ProgressEntryResponse(
                id=entry.id,
                session_id=entry.session_id,
                accuracy_trend=entry.accuracy_trend,
                words_practiced=entry.words_practiced,
                timestamp=entry.timestamp,
            )
            for entry in entries
        ],
        difficult_words=profile.difficult_words if profile else [],
    )


def _reading_level(accuracy_pct: float) -> str:
    if accuracy_pct >= 95:
        return "Advanced Support"
    if accuracy_pct >= 85:
        return "Developing"
    return "Foundational"


def _extract_model_profile_scores(results: dict[str, object]) -> dict[str, float]:
    value = results.get("model_profile_scores", {})
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(score)
        for key, score in value.items()
        if isinstance(score, (int, float))
    }
