import math
import shutil
import subprocess
import tempfile
from typing import TypedDict

import librosa
import numpy as np


class VoiceMetrics(TypedDict):
    speech_rate_wps: float
    pause_duration_ms: float
    pause_frequency: float
    mispronunciation_rate: float
    repetition_rate: float


def extract_voice_metrics(
    spoken_text: str,
    expected_text: str,
    errors: object,
    speed_wpm: float,
    audio_bytes: bytes,
) -> VoiceMetrics:
    words = [token for token in spoken_text.split() if token.strip()]
    expected_word_count = max(len(expected_text.split()), 1)
    audio_signal, sample_rate = _load_audio_signal(audio_bytes)

    duration_seconds = _audio_duration_seconds(audio_signal, sample_rate)
    if duration_seconds <= 0 and speed_wpm > 0:
        duration_seconds = len(words) / (speed_wpm / 60)
    duration_seconds = max(duration_seconds, 1e-6)

    pause_count, pause_duration_ms = _pause_stats(audio_signal, sample_rate)
    if pause_count == 0 and pause_duration_ms == 0:
        # Fallback: derive rough pause signal from NLP hesitation points when raw audio is unavailable.
        pause_count = _estimate_pause_count_from_errors(errors)
        pause_duration_ms = float(pause_count * 350)

    speech_rate_wps = len(words) / duration_seconds
    pause_frequency = pause_count / (duration_seconds / 60) if duration_seconds > 0 else 0.0
    mispronunciation_rate = _mispronunciation_rate(errors, expected_word_count)
    repetition_rate = _repetition_rate(words)

    return {
        "speech_rate_wps": round(max(speech_rate_wps, 0.0), 4),
        "pause_duration_ms": round(max(pause_duration_ms, 0.0), 2),
        "pause_frequency": round(max(pause_frequency, 0.0), 4),
        "mispronunciation_rate": round(min(max(mispronunciation_rate, 0.0), 1.0), 4),
        "repetition_rate": round(min(max(repetition_rate, 0.0), 1.0), 4),
    }


def extract_pitch_variation_hz(audio_bytes: bytes, fallback: float = 28.0) -> float:
    audio_signal, sample_rate = _load_audio_signal(audio_bytes)
    if audio_signal is None or sample_rate is None:
        return fallback
    if audio_signal.size < sample_rate:
        return fallback

    try:
        pitch_track = librosa.yin(
            y=audio_signal,
            fmin=65,
            fmax=450,
            sr=sample_rate,
            frame_length=2048,
            hop_length=512,
        )
    except Exception:
        return fallback

    valid = pitch_track[np.isfinite(pitch_track)]
    if valid.size < 3:
        return fallback

    return round(float(np.clip(np.std(valid), 0.0, 200.0)), 2)


def _audio_duration_seconds(audio_signal: np.ndarray | None, sample_rate: int | None) -> float:
    if audio_signal is None or sample_rate is None or sample_rate <= 0:
        return 0.0
    if audio_signal.size == 0:
        return 0.0
    return float(audio_signal.size / sample_rate)


def _pause_stats(audio_signal: np.ndarray | None, sample_rate: int | None) -> tuple[int, float]:
    if audio_signal is None or sample_rate is None:
        return 0, 0.0
    if audio_signal.size == 0:
        return 0, 0.0

    frame_length = 2048
    hop_length = 512
    try:
        rms = librosa.feature.rms(y=audio_signal, frame_length=frame_length, hop_length=hop_length)[0]
    except Exception:
        return 0, 0.0
    if rms.size == 0:
        return 0, 0.0

    # Adaptive threshold from low-energy region; handles varying microphones/noise floors.
    silence_threshold = float(np.percentile(rms, 20) * 1.3)
    frame_ms = (hop_length / sample_rate) * 1000.0
    min_pause_frames = max(int(math.ceil(200.0 / frame_ms)), 1)  # 200ms minimum pause

    pause_count = 0
    pause_frames = 0
    run = 0
    for frame_rms in rms:
        if frame_rms <= silence_threshold:
            run += 1
        else:
            if run >= min_pause_frames:
                pause_count += 1
                pause_frames += run
            run = 0
    if run >= min_pause_frames:
        pause_count += 1
        pause_frames += run

    return pause_count, pause_frames * frame_ms


def _load_audio_signal(audio_bytes: bytes) -> tuple[np.ndarray | None, int | None]:
    if not audio_bytes:
        return None, None

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        try:
            command = [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                "16000",
                "pipe:1",
            ]
            result = subprocess.run(
                command,
                input=audio_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode == 0 and result.stdout:
                decoded = np.frombuffer(result.stdout, dtype=np.float32)
                if decoded.size > 0:
                    return decoded, 16000
        except Exception:
            pass

    # Fallback path when ffmpeg is not on PATH but librosa backend can decode the file.
    for suffix in (".webm", ".wav", ".mp3", ".m4a"):
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio.flush()
                signal, sample_rate = librosa.load(temp_audio.name, sr=16000, mono=True)
                if signal.size > 0:
                    return signal.astype(np.float32), int(sample_rate)
        except Exception:
            continue

    return None, None


def _estimate_pause_count_from_errors(errors: object) -> int:
    if not isinstance(errors, list):
        return 0
    return min(max(len(errors), 0), 12)


def _mispronunciation_rate(errors: object, expected_word_count: int) -> float:
    if not isinstance(errors, list):
        return 0.0
    mispronunciations = sum(
        1
        for error in errors
        if isinstance(error, dict) and str(error.get("type", "")).lower() == "substitution"
    )
    return mispronunciations / expected_word_count


def _repetition_rate(words: list[str]) -> float:
    if not words:
        return 0.0
    repeated_tokens = 0
    previous = ""
    for word in words:
        cleaned = word.strip(".,!?;:\"'()[]{}").lower()
        if cleaned and cleaned == previous:
            repeated_tokens += 1
        previous = cleaned
    return repeated_tokens / len(words)
