import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import require_role
from app.core.redis import cache_json, delete_cached_value, get_cached_json
from app.models import EyeTrackingFeature, PersonalizedContent, Session, SessionResult, User, VoiceFeature
from app.models.session import SessionStatus, SessionType
from app.schemas.lesson import PersonalizedContentResponse
from app.schemas.session import (
    DiagnosticStartResponse,
    DiagnosticSubmitResponse,
    ReadingStartRequest,
    ReadingStartResponse,
    ReadingSubmitResponse,
    SessionResultPayload,
)
from app.services.content import DIAGNOSTIC_PASSAGE
from app.services.dyslexia_profile_inference import (
    build_profiler_features,
    predict_profile_scores,
)
from app.services.eye_features import extract_eye_tracking_metrics
from app.services.profile import build_profile_response, create_or_update
from app.services.progress import create_progress_entry
from app.services.voice_features import extract_voice_metrics
from app.stubs import eye_tracker, nlp, stt


router = APIRouter()


@router.post("/diagnostic/start", response_model=DiagnosticStartResponse)
async def start_diagnostic(
    current_user: User = Depends(require_role("student")),
    db: AsyncSession = Depends(get_db),
) -> DiagnosticStartResponse:
    session = Session(
        student_id=current_user.id,
        session_type=SessionType.diagnostic,
        status=SessionStatus.active.value,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    await cache_json(
        _session_key(session.id),
        {"expected_text": DIAGNOSTIC_PASSAGE, "session_type": SessionType.diagnostic.value},
    )
    return DiagnosticStartResponse(session_id=session.id, expected_text=DIAGNOSTIC_PASSAGE)


@router.post("/diagnostic/submit", response_model=DiagnosticSubmitResponse)
async def submit_diagnostic(
    session_id: int = Form(...),
    eye_tracking_payload: str = Form(default="{}"),
    audio_file: UploadFile = File(...),
    current_user: User = Depends(require_role("student")),
    db: AsyncSession = Depends(get_db),
) -> DiagnosticSubmitResponse:
    return await _submit_session(
        db=db,
        current_user=current_user,
        session_id=session_id,
        eye_tracking_payload=eye_tracking_payload,
        audio_file=audio_file,
        expected_session_type=SessionType.diagnostic,
        create_progress=False,
    )


@router.post("/reading/start", response_model=ReadingStartResponse)
async def start_reading(
    payload: ReadingStartRequest,
    current_user: User = Depends(require_role("student")),
    db: AsyncSession = Depends(get_db),
) -> ReadingStartResponse:
    result = await db.execute(
        select(PersonalizedContent).where(
            PersonalizedContent.id == payload.personalized_content_id,
            PersonalizedContent.student_id == current_user.id,
        )
    )
    personalized = result.scalar_one_or_none()
    if personalized is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Personalized content not found"
        )

    segments = [str(segment) for segment in personalized.adapted_content.get("segments", [])]
    expected_text = " ".join(segments)

    session = Session(
        student_id=current_user.id,
        session_type=SessionType.reading,
        status=SessionStatus.active.value,
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    await cache_json(
        _session_key(session.id),
        {
            "expected_text": expected_text,
            "session_type": SessionType.reading.value,
            "personalized_content_id": personalized.id,
        },
    )

    return ReadingStartResponse(
        session_id=session.id,
        content=PersonalizedContentResponse(
            id=personalized.id,
            lesson_id=personalized.lesson_id,
            student_id=personalized.student_id,
            segments=segments,
            syllable_breaks=personalized.syllable_breaks,
            font_size=personalized.font_size,
            line_spacing=personalized.spacing,
            chunk_size=personalized.chunk_size,
            created_at=personalized.created_at,
        ),
    )


@router.post("/reading/submit", response_model=ReadingSubmitResponse)
async def submit_reading(
    session_id: int = Form(...),
    eye_tracking_payload: str = Form(default="{}"),
    audio_file: UploadFile = File(...),
    current_user: User = Depends(require_role("student")),
    db: AsyncSession = Depends(get_db),
) -> ReadingSubmitResponse:
    response = await _submit_session(
        db=db,
        current_user=current_user,
        session_id=session_id,
        eye_tracking_payload=eye_tracking_payload,
        audio_file=audio_file,
        expected_session_type=SessionType.reading,
        create_progress=True,
    )
    return ReadingSubmitResponse(
        result=response["result"],
        profile=response["profile"],
        progress_entry_id=response["progress_entry_id"],
    )


async def _submit_session(
    db: AsyncSession,
    current_user: User,
    session_id: int,
    eye_tracking_payload: str,
    audio_file: UploadFile,
    expected_session_type: SessionType,
    create_progress: bool,
) -> DiagnosticSubmitResponse | dict[str, object]:
    payload = await get_cached_json(_session_key(session_id))
    if payload is None:
        if expected_session_type == SessionType.diagnostic:
            # Diagnostic sessions can recover after backend restarts because the expected text is static.
            payload = {
                "expected_text": DIAGNOSTIC_PASSAGE,
                "session_type": SessionType.diagnostic.value,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session cache not found. Start a new reading session and submit again.",
            )

    session_result = await db.execute(
        select(Session).where(Session.id == session_id, Session.student_id == current_user.id)
    )
    session = session_result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.session_type != expected_session_type:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session type")

    existing_result = await db.execute(select(SessionResult).where(SessionResult.session_id == session_id))
    existing = existing_result.scalar_one_or_none()
    if existing is not None:
        if expected_session_type == SessionType.diagnostic:
            profile = await build_profile_response(db, current_user.id)
            return DiagnosticSubmitResponse(
                result=_result_payload_from_existing(session, existing),
                profile=profile,
            )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Session already submitted")

    eye_payload = _parse_eye_payload(eye_tracking_payload)

    expected_text = str(payload["expected_text"])
    stt.prime_expected_text(expected_text)
    audio_bytes = await audio_file.read()
    spoken_text = await stt.transcribe(audio_bytes)
    nlp_result = await nlp.compare_texts(spoken_text, expected_text)
    eye_result = await eye_tracker.analyze(eye_payload)
    accuracy_pct = _accuracy_pct(expected_text, nlp_result["errors"])

    result_model = SessionResult(
        session_id=session.id,
        spoken_text=spoken_text,
        expected_text=expected_text,
        errors=nlp_result["errors"],
        speed_wpm=float(nlp_result["speed_wpm"]),
        hesitation_points=nlp_result["hesitation_points"],
        eye_tracking_data=eye_result,
        accuracy_pct=accuracy_pct,
    )
    db.add(result_model)
    focus_events = eye_payload.get("focus_events", []) if isinstance(eye_payload, dict) else []
    metrics = extract_eye_tracking_metrics(
        focus_events=focus_events if isinstance(focus_events, list) else [],
        expected_word_count=len(expected_text.split()),
    )
    db.add(
        EyeTrackingFeature(
            student_id=current_user.id,
            session_id=session.id,
            fixation_duration_ms=metrics["fixation_duration_ms"],
            saccade_length=metrics["saccade_length"],
            regression_count=metrics["regression_count"],
            skipped_words=metrics["skipped_words"],
            reading_speed_wpm=metrics["reading_speed_wpm"],
        )
    )
    voice_metrics = extract_voice_metrics(
        spoken_text=spoken_text,
        expected_text=expected_text,
        errors=nlp_result["errors"],
        speed_wpm=float(nlp_result["speed_wpm"]),
        audio_bytes=audio_bytes,
    )
    db.add(
        VoiceFeature(
            student_id=current_user.id,
            session_id=session.id,
            speech_rate_wps=voice_metrics["speech_rate_wps"],
            pause_duration_ms=voice_metrics["pause_duration_ms"],
            pause_frequency=voice_metrics["pause_frequency"],
            mispronunciation_rate=voice_metrics["mispronunciation_rate"],
            repetition_rate=voice_metrics["repetition_rate"],
        )
    )
    profiler_features = build_profiler_features(
        eye_metrics=metrics,
        voice_metrics=voice_metrics,
        expected_word_count=len(expected_text.split()),
        audio_bytes=audio_bytes,
    )
    model_profile_scores = predict_profile_scores(profiler_features) or {}

    session.status = SessionStatus.completed.value
    session.ended_at = datetime.now(timezone.utc)

    await create_or_update(
        db,
        current_user.id,
        {
            "errors": nlp_result["errors"],
            "speed_wpm": nlp_result["speed_wpm"],
            "accuracy_pct": accuracy_pct,
            "attention_score": eye_result["attention_score"],
            "model_profile_scores": model_profile_scores,
        },
    )

    progress_entry_id: int | None = None
    if create_progress:
        progress_entry = await create_progress_entry(
            db=db,
            student_id=current_user.id,
            session_id=session.id,
            accuracy_trend=accuracy_pct,
            words_practiced=_words_practiced(expected_text, nlp_result["errors"]),
        )
        progress_entry_id = progress_entry.id

    await db.commit()
    await delete_cached_value(_session_key(session_id))

    profile = await build_profile_response(db, current_user.id)
    result_payload = SessionResultPayload(
        session_id=session.id,
        spoken_text=spoken_text,
        expected_text=expected_text,
        errors=nlp_result["errors"],
        speed_wpm=float(nlp_result["speed_wpm"]),
        hesitation_points=list(nlp_result["hesitation_points"]),
        attention_score=float(eye_result["attention_score"]),
        skip_events=list(eye_result["skip_events"]),
        re_read_events=list(eye_result["re_read_events"]),
        avg_fixation_ms=int(eye_result["avg_fixation_ms"]),
        accuracy_pct=accuracy_pct,
    )

    if create_progress:
        return {"result": result_payload, "profile": profile, "progress_entry_id": progress_entry_id}
    return DiagnosticSubmitResponse(result=result_payload, profile=profile)


def _accuracy_pct(expected_text: str, errors: object) -> float:
    expected_words = max(len(expected_text.split()), 1)
    error_count = len(errors) if isinstance(errors, list) else 0
    return round(max(0.0, ((expected_words - error_count) / expected_words) * 100), 2)


def _words_practiced(expected_text: str, errors: object) -> list[str]:
    error_words = [
        str(item["word"])
        for item in errors
        if isinstance(item, dict) and item.get("word")
    ] if isinstance(errors, list) else []
    return error_words or expected_text.split()[:5]


def _session_key(session_id: int) -> str:
    return f"readable:session:{session_id}"


def _parse_eye_payload(eye_tracking_payload: str) -> dict[str, object]:
    raw = (eye_tracking_payload or "").strip()
    if not raw or raw.lower() in {"undefined", "null"}:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _result_payload_from_existing(session: Session, existing: SessionResult) -> SessionResultPayload:
    eye_data = existing.eye_tracking_data if isinstance(existing.eye_tracking_data, dict) else {}
    return SessionResultPayload(
        session_id=session.id,
        spoken_text=existing.spoken_text,
        expected_text=existing.expected_text,
        errors=existing.errors if isinstance(existing.errors, list) else [],
        speed_wpm=float(existing.speed_wpm),
        hesitation_points=(
            list(existing.hesitation_points) if isinstance(existing.hesitation_points, list) else []
        ),
        attention_score=float(eye_data.get("attention_score", 0.0)),
        skip_events=list(eye_data.get("skip_events", []))
        if isinstance(eye_data.get("skip_events", []), list)
        else [],
        re_read_events=list(eye_data.get("re_read_events", []))
        if isinstance(eye_data.get("re_read_events", []), list)
        else [],
        avg_fixation_ms=int(eye_data.get("avg_fixation_ms", 0) or 0),
        accuracy_pct=float(existing.accuracy_pct),
    )
