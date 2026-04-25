from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.voice_features import extract_pitch_variation_hz


MODEL_PATH = Path(__file__).resolve().parents[2] / "profile_model" / "dyslexia_profiler.pt"


def predict_profile_scores(features: dict[str, float]) -> dict[str, float] | None:
    if not MODEL_PATH.exists():
        return None

    try:
        import importlib.util
        import sys

        model_file = Path(__file__).resolve().parents[2] / "profile_model" / "model.py"
        spec = importlib.util.spec_from_file_location("profile_model_impl", model_file)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules["profile_model_impl"] = module
        spec.loader.exec_module(module)

        import numpy as np
        import torch

        profiler_cls = getattr(module, "DyslexiaProfiler", None)
        feature_names = getattr(module, "FEATURE_NAMES", None)
        output_names = getattr(module, "OUTPUT_NAMES", None)
        feature_bounds = getattr(module, "FEATURE_BOUNDS", None)
        if profiler_cls is None or feature_names is None:
            return None
        if output_names is None:
            return None

        model, scaler = profiler_cls.load(str(MODEL_PATH))
        arr = np.array([float(features.get(name, 0.0)) for name in feature_names], dtype=np.float32)

        if scaler and isinstance(scaler, dict) and "mean_" in scaler and "scale_" in scaler:
            mean = np.array(scaler["mean_"], dtype=np.float32)
            scale = np.array(scaler["scale_"], dtype=np.float32)
            arr = (arr - mean) / (scale + 1e-8)
        elif isinstance(feature_bounds, dict):
            for index, name in enumerate(feature_names):
                bounds = feature_bounds.get(name)
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    continue
                low, high = float(bounds[0]), float(bounds[1])
                arr[index] = np.clip((arr[index] - low) / (high - low + 1e-8), 0.0, 1.0)

        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model.forward(x)
        return {name: float(out[name].item()) for name in output_names if name in out}
    except Exception:
        return None


def build_profiler_features(
    *,
    eye_metrics: dict[str, Any],
    voice_metrics: dict[str, Any],
    expected_word_count: int,
    audio_bytes: bytes,
) -> dict[str, float]:
    skipped_words = float(eye_metrics.get("skipped_words", 0.0))
    skipped_word_rate = skipped_words / max(float(expected_word_count), 1.0)

    return {
        "fixation_duration_ms": float(eye_metrics.get("fixation_duration_ms", 0.0)),
        "saccade_length_deg": float(eye_metrics.get("saccade_length", 0.0)),
        "regression_count": float(eye_metrics.get("regression_count", 0.0)),
        "skipped_word_rate": max(0.0, min(skipped_word_rate, 1.0)),
        "reading_speed_wpm": float(eye_metrics.get("reading_speed_wpm", 0.0)),
        "speech_rate_wps": float(voice_metrics.get("speech_rate_wps", 0.0)),
        "pause_duration_ms": float(voice_metrics.get("pause_duration_ms", 0.0)),
        "pause_frequency": float(voice_metrics.get("pause_frequency", 0.0)),
        "mispronunciation_rate": float(voice_metrics.get("mispronunciation_rate", 0.0)),
        "repetition_rate": float(voice_metrics.get("repetition_rate", 0.0)),
        "pitch_variation_hz": extract_pitch_variation_hz(audio_bytes),
    }
