"""
predict.py
==========
Load a trained DyslexiaProfiler and run inference.

Modes
-----
1. Single prediction (dict of feature values)
2. Batch prediction from CSV
3. Interactive CLI — type in features, get profile

Usage
-----
# Interactive
python predict.py --model checkpoints/dyslexia_profiler.pt

# Single JSON
python predict.py \
    --model checkpoints/dyslexia_profiler.pt \
    --features '{"fixation_duration_ms": 320, "saccade_length_deg": 3.5, ...}'

# Batch CSV
python predict.py \
    --model checkpoints/dyslexia_profiler.pt \
    --csv data/new_participants.csv \
    --out results/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from model import DyslexiaProfiler, FEATURE_NAMES, OUTPUT_NAMES, FEATURE_BOUNDS
from data_generator import csv_to_samples, RawFeatures


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE INTERPRETER
# ─────────────────────────────────────────────────────────────────────────────

# Severity thresholds (tuned on population averages from literature)
SEVERITY_BANDS = [
    (0.00, 0.25, "Minimal",  "✅"),
    (0.25, 0.45, "Mild",     "🟡"),
    (0.45, 0.65, "Moderate", "🟠"),
    (0.65, 0.85, "High",     "🔴"),
    (0.85, 1.01, "Severe",   "🆘"),
]

# For reading_fluency the scale is INVERTED (1=good, 0=poor)
FLUENCY_BANDS = [
    (0.75, 1.01, "Strong",   "✅"),
    (0.55, 0.75, "Adequate", "🟡"),
    (0.35, 0.55, "Below Avg","🟠"),
    (0.15, 0.35, "Poor",     "🔴"),
    (0.00, 0.15, "Very Poor","🆘"),
]


def _band(score: float, bands) -> tuple:
    for lo, hi, label, icon in bands:
        if lo <= score < hi:
            return label, icon
    return "Unknown", "❓"


def _bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)


PROFILE_DESCRIPTIONS = {
    "reading_fluency": {
        "title": "Reading Fluency",
        "desc": "Overall smoothness and speed of reading. "
                "Combines speech rate, pause patterns, and repetitions.",
        "invert": True,   # Higher score = better
    },
    "decoding_difficulty": {
        "title": "Decoding Difficulty",
        "desc": "Effort required to decode written words. "
                "Elevated by mispronunciations and long pauses.",
        "invert": False,
    },
    "phonological_difficulty": {
        "title": "Phonological Difficulty",
        "desc": "Difficulty mapping letters to sounds. "
                "Key marker of classic dyslexia.",
        "invert": False,
    },
    "visual_difficulty": {
        "title": "Visual Processing Difficulty",
        "desc": "Issues with visual text scanning. "
                "Indicated by high regressions and fixation duration.",
        "invert": False,
    },
    "attention_difficulty": {
        "title": "Attention / Consistency Difficulty",
        "desc": "Inconsistency in gaze and speech patterns, "
                "possibly linked to attention challenges.",
        "invert": False,
    },
}

INTERVENTION_MAP = {
    "decoding_difficulty":     [
        "Structured phonics programs (Orton-Gillingham, Wilson Reading)",
        "Decodable text practice at appropriate level",
        "Multi-sensory letter-sound mapping activities",
    ],
    "phonological_difficulty": [
        "Phoneme awareness training (blending, segmenting)",
        "Rhyming and alliteration games",
        "Explicit syllable decomposition exercises",
    ],
    "visual_difficulty":       [
        "Coloured overlays / tinted lenses assessment",
        "Larger font size and increased line spacing",
        "Saccadic training with optometrist",
        "Text-to-speech tools to reduce visual load",
    ],
    "attention_difficulty":    [
        "Structured reading environment (minimal distractions)",
        "Metacognitive reading strategies",
        "Short, timed reading intervals with breaks",
        "ADHD screening referral if persistent",
    ],
}


def build_profile_report(scores: Dict[str, float],
                          subject_id: str = "Unknown") -> str:
    """Generate a human-readable profile report."""

    lines = []
    lines.append("=" * 60)
    lines.append(f"  DYSLEXIA PROFILE REPORT  —  Subject: {subject_id}")
    lines.append("=" * 60)

    # Overall dyslexia risk score (weighted average of difficulty scores)
    difficulty_scores = [
        scores["decoding_difficulty"],
        scores["phonological_difficulty"],
        scores["visual_difficulty"],
        scores["attention_difficulty"],
        1.0 - scores["reading_fluency"],   # invert fluency
    ]
    overall_risk = float(np.mean(difficulty_scores))
    risk_label, risk_icon = _band(overall_risk, SEVERITY_BANDS)

    lines.append(f"\n  Overall Dyslexia Risk : {risk_icon} {risk_label} "
                 f"({overall_risk:.2f})  {_bar(overall_risk)}")
    lines.append("")
    lines.append("  ── Individual Profile Scores ─────────────────────────")

    for name in OUTPUT_NAMES:
        score = scores[name]
        meta = PROFILE_DESCRIPTIONS[name]
        invert = meta["invert"]

        if invert:
            display_score = score
            label, icon = _band(score, FLUENCY_BANDS)
        else:
            display_score = score
            label, icon = _band(score, SEVERITY_BANDS)

        bar = _bar(display_score if not invert else 1 - display_score)
        lines.append(f"\n  {icon} {meta['title']}")
        lines.append(f"     Score  : {score:.3f}  [{bar}]  {label}")
        lines.append(f"     Detail : {meta['desc']}")

    # Interventions for elevated dimensions
    elevated = [
        name for name in OUTPUT_NAMES
        if name != "reading_fluency" and scores[name] >= 0.45
    ]
    if scores["reading_fluency"] < 0.55:
        elevated.insert(0, "decoding_difficulty")  # treat as decoding concern

    if elevated:
        lines.append("\n  ── Recommended Interventions ────────────────────────")
        seen = set()
        for dim in elevated:
            key = dim
            if key in INTERVENTION_MAP and key not in seen:
                seen.add(key)
                dim_title = PROFILE_DESCRIPTIONS[dim]["title"]
                lines.append(f"\n  [{dim_title}]")
                for tip in INTERVENTION_MAP[key]:
                    lines.append(f"    • {tip}")
    else:
        lines.append("\n  ✅ No significant difficulties detected.")
        lines.append("     Continue monitoring with periodic reassessment.")

    lines.append("\n" + "=" * 60)
    lines.append("  Note: This tool is a screening aid, not a clinical diagnosis.")
    lines.append("  Consult an educational psychologist for formal assessment.")
    lines.append("=" * 60)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_scaler(model_path: str):
    """Load trained model and scaler parameters."""
    model, scaler_params = DyslexiaProfiler.load(model_path)
    model.eval()
    print(f"[✓] Model loaded from {model_path}")
    if scaler_params:
        print(f"    Features: {len(scaler_params['feature_names'])}")
    return model, scaler_params


def predict_from_dict(
    model: DyslexiaProfiler,
    raw_features: dict,
    scaler_params: Optional[dict] = None,
) -> Dict[str, float]:
    """Predict profile from a dict of raw feature values."""
    return model.predict_single(raw_features, scaler_params)


def predict_batch_csv(
    model: DyslexiaProfiler,
    csv_path: str,
    out_path: str,
    scaler_params: Optional[dict] = None,
):
    """Load CSV, predict for each row, save results."""
    samples = csv_to_samples(csv_path)
    print(f"[✓] Loaded {len(samples)} samples from {csv_path}")

    results = []
    for s in samples:
        raw = asdict(s.features)
        scores = predict_from_dict(model, raw, scaler_params)
        row = {"subject_id": s.subject_id, "is_dyslexic_gt": s.is_dyslexic}
        row.update({f"pred_{k}": round(v, 4) for k, v in scores.items()})
        row.update({f"true_{k}": round(getattr(s.labels, k), 4) for k in OUTPUT_NAMES})
        results.append(row)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"[✓] Predictions saved → {out_path}")
    _print_batch_summary(results)


def _print_batch_summary(results: list):
    """Print summary statistics for batch predictions."""
    import numpy as np
    print("\n── Batch Prediction Summary ──────────────────────────────")
    for name in OUTPUT_NAMES:
        vals = [r[f"pred_{name}"] for r in results]
        print(f"  {name:<28}  "
              f"mean={np.mean(vals):.3f}  "
              f"std={np.std(vals):.3f}  "
              f"max={np.max(vals):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE CLI
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_PROMPTS = {
    "fixation_duration_ms":    ("Mean fixation duration",    "ms",      220,  80,   600),
    "saccade_length_deg":      ("Mean saccade amplitude",    "degrees", 5.5,  0.5,  15),
    "regression_count":        ("Regressions per 100 words", "",        8,    0,    60),
    "skipped_word_rate":       ("Skipped word rate",         "0-1",     0.08, 0.0,  0.6),
    "reading_speed_wpm":       ("Reading speed",             "WPM",     240,  30,   500),
    "speech_rate_wps":         ("Speech rate",               "words/s", 2.8,  0.3,  5.0),
    "pause_duration_ms":       ("Mean pause duration",       "ms",      180,  50,   1200),
    "pause_frequency":         ("Pauses per minute",         "",        6,    0,    40),
    "mispronunciation_rate":   ("Mispronunciation rate",     "0-1",     0.03, 0.0,  0.6),
    "repetition_rate":         ("Word repetition rate",      "0-1",     0.02, 0.0,  0.5),
    "pitch_variation_hz":      ("Pitch variation (std)",     "Hz",      28,   5,    100),
}


def interactive_session(model: DyslexiaProfiler,
                         scaler_params: Optional[dict]):
    """Interactive feature entry and profile display."""
    print("\n" + "=" * 60)
    print("  DYSLEXIA PROFILER — Interactive Mode")
    print("  Press Enter to use defaults (typical adult reader)")
    print("=" * 60)

    subject_id = input("\n  Subject ID [ANON]: ").strip() or "ANON"

    raw = {}
    print("\n── Eye-Tracking Features ────────────────────────────────")
    eye_names = FEATURE_NAMES[:5]
    for name in eye_names:
        label, unit, default, lo, hi = FEATURE_PROMPTS[name]
        unit_str = f" ({unit})" if unit else ""
        while True:
            try:
                val = input(f"  {label}{unit_str} [{default}]: ").strip()
                val = float(val) if val else default
                if lo <= val <= hi:
                    raw[name] = val
                    break
                print(f"    ⚠ Enter a value between {lo} and {hi}")
            except ValueError:
                print("    ⚠ Please enter a number")

    print("\n── Speech Features ──────────────────────────────────────")
    speech_names = FEATURE_NAMES[5:10]
    for name in speech_names:
        label, unit, default, lo, hi = FEATURE_PROMPTS[name]
        unit_str = f" ({unit})" if unit else ""
        while True:
            try:
                val = input(f"  {label}{unit_str} [{default}]: ").strip()
                val = float(val) if val else default
                if lo <= val <= hi:
                    raw[name] = val
                    break
                print(f"    ⚠ Enter a value between {lo} and {hi}")
            except ValueError:
                print("    ⚠ Please enter a number")

    print("\n── Acoustic Features ────────────────────────────────────")
    for name in FEATURE_NAMES[10:]:
        label, unit, default, lo, hi = FEATURE_PROMPTS[name]
        unit_str = f" ({unit})" if unit else ""
        while True:
            try:
                val = input(f"  {label}{unit_str} [{default}]: ").strip()
                val = float(val) if val else default
                if lo <= val <= hi:
                    raw[name] = val
                    break
                print(f"    ⚠ Enter a value between {lo} and {hi}")
            except ValueError:
                print("    ⚠ Please enter a number")

    print("\n[⚙] Running profile inference...")
    scores = predict_from_dict(model, raw, scaler_params)
    report = build_profile_report(scores, subject_id=subject_id)
    print("\n" + report)

    # Ask to save
    save = input("\n  Save report to file? (y/n) [n]: ").strip().lower()
    if save == "y":
        fname = f"report_{subject_id.replace(' ', '_')}.txt"
        with open(fname, "w") as f:
            f.write(report)
            f.write("\n\nRAW SCORES (JSON):\n")
            f.write(json.dumps(scores, indent=2))
        print(f"  [✓] Saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DyslexiaProfiler — Inference")
    parser.add_argument("--model", type=str, default="checkpoints/dyslexia_profiler.pt",
                        help="Path to trained .pt model file")
    parser.add_argument("--features", type=str, default=None,
                        help="JSON string of raw feature values")
    parser.add_argument("--csv", type=str, default=None,
                        help="Input CSV for batch prediction")
    parser.add_argument("--out", type=str, default="results/predictions.csv",
                        help="Output CSV path for batch mode")
    parser.add_argument("--subject", type=str, default="ANON",
                        help="Subject ID for single prediction")
    args = parser.parse_args()

    # Load model
    if not Path(args.model).exists():
        print(f"[✗] Model not found at {args.model}")
        print("    Run: python train.py --data data/dataset.csv --out checkpoints/")
        sys.exit(1)

    model, scaler_params = load_model_and_scaler(args.model)

    if args.features:
        # Single prediction from JSON
        raw = json.loads(args.features)
        # Fill missing features with defaults
        for name in FEATURE_NAMES:
            if name not in raw:
                _, _, default, _, _ = FEATURE_PROMPTS[name]
                raw[name] = default
                print(f"  [!] Missing {name} — using default {default}")

        scores = predict_from_dict(model, raw, scaler_params)
        report = build_profile_report(scores, subject_id=args.subject)
        print(report)

    elif args.csv:
        # Batch mode
        predict_batch_csv(model, args.csv, args.out, scaler_params)

    else:
        # Interactive mode
        while True:
            interactive_session(model, scaler_params)
            again = input("\n  Run another session? (y/n) [n]: ").strip().lower()
            if again != "y":
                break


if __name__ == "__main__":
    main()